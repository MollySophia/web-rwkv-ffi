use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    ffi::{c_char, CStr},
    mem::size_of,
    mem::ManuallyDrop,
    mem::MaybeUninit,
    path::Path,
    sync::{Arc, Mutex, OnceLock, RwLock},
    time::Duration,
};

use anyhow::Result;
use half::{bf16, f16, slice::HalfFloatSliceExt};
use itertools::Itertools;
use memmap2::Mmap;
#[cfg(unix)]
use memmap2::UncheckedAdvice;
use ops::TensorOpExt;
use rayon::{
    prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut},
    ThreadPool, ThreadPoolBuilder,
};
use repugnant_pickle::{RepugnantTorchTensors as TorchTensors, TensorType};
use safetensors::View;
use safetensors::{Dtype, SafeTensors};
use serde::{de::DeserializeSeed, Deserialize};
#[cfg(windows)]
use std::ffi::c_void;
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    num::Float,
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption, Token},
        loader::{Loader, Reader},
        model::{Bundle, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant, State},
        softmax::softmax_one,
        v4, v5, v6, v7, TokioRuntime,
    },
    tensor::{ops::TensorOp, serialization::Seed},
    wgpu,
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

mod ops;

static RUNTIME: RwLock<Option<WktvRuntime>> = RwLock::new(None);
static CONVERT_THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();

#[derive(Clone)]
struct WktvRuntime {
    runtime: TokioRuntime<Rnn>,
    info: ModelInfo,
    state: Arc<dyn State + Sync + Send + 'static>,
    context: Context,
    tokio: Arc<tokio::runtime::Runtime>,
}

fn make_hooks_extended_v6<F: Float>(info: &ModelInfo) -> Result<v6::HookMap<F>> {
    let mut hooks = v6::HookMap::new();
    for layer in 0..info.num_layer {
        // add a custom operation before time-mix for each layer
        hooks.insert(
            v6::Hook::PreAttTimeDecayActivate(layer),
            Box::new(move |frame: v6::Frame<F>| {
                let op = TensorOp::ext_v6(&frame.buffer.time_decay, &frame.buffer.att_k)?;
                Ok(TensorOp::List(vec![op]))
            }),
        );
    }
    Ok(hooks)
}

fn make_hooks_extended_v7<F: Float>(info: &ModelInfo) -> Result<v7::HookMap<F>> {
    let mut hooks = v7::HookMap::new();
    for layer in 0..info.num_layer {
        hooks.insert(
            v7::Hook::PostAttAdapt(layer),
            Box::new(move |frame: v7::Frame<F>| {
                let op = TensorOp::affine(&frame.buffer.att_a, 2.0, 0.0)?;
                Ok(TensorOp::List(vec![op]))
            }),
        );
        hooks.insert(
            v7::Hook::PostAttControl(layer),
            Box::new(move |frame: v7::Frame<F>| {
                let op = TensorOp::ext_v7(&frame.buffer.att_w, &frame.buffer.att_a)?;
                Ok(TensorOp::List(vec![op]))
            }),
        );
    }
    Ok(hooks)
}

#[derive(Debug, Deserialize)]
struct Prefab {
    info: ModelInfo,
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

#[cfg(windows)]
type WinHandle = *mut c_void;

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentProcess() -> WinHandle;
    fn K32EmptyWorkingSet(handle: WinHandle) -> i32;
}

fn trim_process_working_set() {
    #[cfg(windows)]
    unsafe {
        let _ = K32EmptyWorkingSet(GetCurrentProcess());
    }
}

// async fn load_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
//     let file = File::open(path).await?;
//     let mut reader = BufReader::new(file);
//     let mut contents = String::new();
//     reader.read_to_string(&mut contents).await?;
//     Ok(Tokenizer::new(&contents)?)
// }

fn load_runtime(
    model: impl AsRef<Path>,
    quant: usize,
    quant_nf4: usize,
    quant_sf4: usize,
    rescale: Option<usize>,
    extended: bool,
    fp16: bool,
    batch: usize,
) -> Result<WktvRuntime> {
    let tokio = Arc::new(tokio::runtime::Runtime::new()?);
    let _tokio = tokio.clone();

    _tokio.block_on(async move {
        let file = File::open(model).await?;
        let data = unsafe { Mmap::map(&file)? };

        let model = SafeTensors::deserialize(&data)?;
        let info = Loader::info(&model)?;
        log::info!("{:#?}", info);

        let context = create_context(&info).await?;
        log::info!("{:#?}", context.adapter.get_info());

        let quant = (0..quant)
            .map(|layer| (layer, Quant::Int8))
            .chain((0..quant_nf4).map(|layer| (layer, Quant::NF4)))
            .chain((0..quant_sf4).map(|layer| (layer, Quant::SF4)))
            .collect();

        let builder = ModelBuilder::new(&context, model).quant(quant);
        let builder = match rescale {
            Some(rescale) => builder.rescale(rescale),
            None => builder,
        };
        let runtime = match info.version {
            ModelVersion::V4 => {
                if fp16 {
                    let model = builder.build_v4().await?;
                    let bundle = v4::Bundle::<f16>::new(model, batch);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let model = builder.build_v4().await?;
                    let bundle = v4::Bundle::<f32>::new(model, batch);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
            ModelVersion::V5 => {
                if fp16 {
                    let model = builder.build_v5().await?;
                    let bundle = v5::Bundle::<f16>::new(model, batch);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let model = builder.build_v5().await?;
                    let bundle = v5::Bundle::<f32>::new(model, batch);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
            ModelVersion::V6 => {
                if fp16 {
                    let model = builder.build_v6().await?;
                    let bundle = match extended {
                        true => {
                            let hooks = make_hooks_extended_v6(&info)?;
                            v6::Bundle::<f16>::new_with_hooks(model, batch, hooks)
                        }
                        false => v6::Bundle::<f16>::new(model, batch),
                    };
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let model = builder.build_v6().await?;
                    let bundle = match extended {
                        true => {
                            let hooks = make_hooks_extended_v6(&info)?;
                            v6::Bundle::<f32>::new_with_hooks(model, batch, hooks)
                        }
                        false => v6::Bundle::<f32>::new(model, batch),
                    };
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
            ModelVersion::V7 => {
                if fp16 {
                    let model = builder.build_v7().await?;
                    let bundle = match extended {
                        true => {
                            let hooks = make_hooks_extended_v7(&info)?;
                            v7::Bundle::<f16>::new_with_hooks(model, batch, hooks)
                        }
                        false => v7::Bundle::<f16>::new(model, batch),
                    };
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let model = builder.build_v7().await?;
                    let bundle = match extended {
                        true => {
                            let hooks = make_hooks_extended_v7(&info)?;
                            v7::Bundle::<f32>::new_with_hooks(model, batch, hooks)
                        }
                        false => v7::Bundle::<f32>::new(model, batch),
                    };
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
        };
        Ok(runtime)
    })
}

fn load_runtime_prefab(model: impl AsRef<Path>, fp16: bool, batch: usize) -> Result<WktvRuntime> {
    let tokio = Arc::new(tokio::runtime::Runtime::new()?);
    let _tokio = tokio.clone();

    _tokio.block_on(async move {
        let file = File::open(model).await?;
        let data = unsafe { Mmap::map(&file)? };

        let Prefab { info } = cbor4ii::serde::from_slice::<Prefab>(&data)?;

        let reader = cbor4ii::core::utils::SliceReader::new(&data);
        let mut deserializer = cbor4ii::serde::Deserializer::new(reader);

        log::info!("{:#?}", info);
        let context = create_context(&info).await?;

        let runtime = match info.version {
            ModelVersion::V4 => {
                if fp16 {
                    let seed: Seed<_, v4::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v4::Bundle::<f16>::new(model, batch);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let seed: Seed<_, v4::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v4::Bundle::<f32>::new(model, batch);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
            ModelVersion::V5 => {
                if fp16 {
                    let seed: Seed<_, v5::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v5::Bundle::<f16>::new(model, batch);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let seed: Seed<_, v5::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v5::Bundle::<f32>::new(model, batch);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
            ModelVersion::V6 => {
                if fp16 {
                    let seed: Seed<_, v6::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v6::Bundle::<f16>::new(model, batch);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let seed: Seed<_, v6::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v6::Bundle::<f32>::new(model, batch);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
            ModelVersion::V7 => {
                if fp16 {
                    let seed: Seed<_, v7::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v7::Bundle::<f16>::new(model, batch);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let seed: Seed<_, v7::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v7::Bundle::<f32>::new(model, batch);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
        };
        Ok(runtime)
    })
}

#[derive(Clone)]
struct TorchTensorMeta {
    source_type: TensorType,
    start: usize,
    end: usize,
    source_shape: Vec<usize>,
    shape: Vec<usize>,
    transpose: bool,
}

#[derive(Clone)]
enum TorchStorage {
    #[cfg(unix)]
    Mmap(Arc<Mmap>),
    #[cfg(not(unix))]
    File(Arc<std::fs::File>),
}

impl TorchStorage {
    fn new(model: impl AsRef<Path>) -> Result<Self> {
        let file = Arc::new(std::fs::File::open(model.as_ref())?);
        #[cfg(unix)]
        {
            let mmap = Arc::new(unsafe { Mmap::map(file.as_ref())? });
            Ok(Self::Mmap(mmap))
        }
        #[cfg(not(unix))]
        {
            Ok(Self::File(file))
        }
    }

    fn read<'a>(
        &'a self,
        start: usize,
        end: usize,
    ) -> Result<Cow<'a, [u8]>, safetensors::SafeTensorError> {
        match self {
            #[cfg(unix)]
            Self::Mmap(mmap) => mmap
                .get(start..end)
                .map(Cow::Borrowed)
                .ok_or(safetensors::SafeTensorError::TensorInvalidInfo),
            #[cfg(not(unix))]
            Self::File(file) => {
                let len = end
                    .checked_sub(start)
                    .ok_or(safetensors::SafeTensorError::TensorInvalidInfo)?;
                let mmap = unsafe {
                    memmap2::MmapOptions::new()
                        .offset(start as u64)
                        .len(len)
                        .map(file.as_ref())?
                };
                Ok(Cow::Owned(mmap[..].to_vec()))
            }
        }
    }

    fn release(&self, start: usize, end: usize) {
        #[cfg(unix)]
        {
            let Self::Mmap(mmap) = self;
            let _ = unsafe {
                mmap.unchecked_advise_range(
                    UncheckedAdvice::DontNeed,
                    start,
                    end.saturating_sub(start),
                )
            };
        }

        #[cfg(not(unix))]
        {
            let _ = (start, end);
        }
    }
}

impl TorchTensorMeta {
    fn data_len(&self) -> usize {
        self.shape.iter().product::<usize>() * size_of::<f16>()
    }

    fn load<'a>(
        &self,
        storage: &'a TorchStorage,
    ) -> Result<Cow<'a, [u8]>, safetensors::SafeTensorError> {
        let bytes = storage.read(self.start, self.end)?;
        match (&self.source_type, self.transpose, bytes) {
            (TensorType::Float16, false, bytes) => Ok(bytes),
            (_, _, bytes) => {
                let data = self.load_f16(bytes.as_ref())?;
                storage.release(self.start, self.end);
                Ok(Cow::Owned(f16_vec_into_bytes(data)))
            }
        }
    }

    fn load_f16(&self, data: &[u8]) -> Result<Vec<f16>, safetensors::SafeTensorError> {
        let invalid = || {
            safetensors::SafeTensorError::InvalidTensorView(
                Dtype::F16,
                self.shape.clone(),
                data.len(),
            )
        };

        match self.source_type {
            TensorType::Float16 => {
                let src: &[f16] = bytemuck::try_cast_slice(data).map_err(|_| invalid())?;
                if self.transpose {
                    transpose_last_two(&self.source_shape, |index| src[index])
                } else {
                    Ok(src.to_vec())
                }
            }
            TensorType::BFloat16 => {
                let src: &[bf16] = bytemuck::try_cast_slice(data).map_err(|_| invalid())?;
                if self.transpose {
                    transpose_last_two(&self.source_shape, |index| {
                        f16::from_f32(src[index].to_f32())
                    })
                } else {
                    Ok(parallel_convert_bf16_to_f16(src))
                }
            }
            TensorType::Float32 => {
                let src: &[f32] = bytemuck::try_cast_slice(data).map_err(|_| invalid())?;
                if self.transpose {
                    transpose_last_two(&self.source_shape, |index| f16::from_f32(src[index]))
                } else {
                    Ok(parallel_convert_f32_to_f16(src))
                }
            }
            _ => Err(safetensors::SafeTensorError::TensorInvalidInfo),
        }
    }
}

struct TorchLoadProgress {
    callback: extern "C" fn(f32),
    total: usize,
    seen: Mutex<HashSet<String>>,
}

impl TorchLoadProgress {
    fn report(&self, name: &str) {
        let mut seen = self
            .seen
            .lock()
            .expect("torch load progress mutex poisoned");
        if seen.insert(name.to_string()) {
            let progress = seen.len() as f32 / self.total as f32 * 0.5;
            (self.callback)(progress);
        }
    }
}

#[derive(Default)]
struct TorchTensorStats {
    calls: usize,
    bytes: usize,
    direct_calls: usize,
    direct_bytes: usize,
    convert_calls: usize,
    convert_bytes: usize,
    convert_f32_calls: usize,
    convert_f32_bytes: usize,
    convert_bf16_calls: usize,
    convert_bf16_bytes: usize,
    transpose_calls: usize,
    transpose_bytes: usize,
    elapsed: Duration,
}

struct TorchReader {
    storage: Arc<TorchStorage>,
    order: Vec<String>,
    tensors: HashMap<String, TorchTensorMeta>,
    progress: Option<TorchLoadProgress>,
    stats: Option<Arc<Mutex<TorchTensorStats>>>,
}

impl TorchReader {
    fn new(model: impl AsRef<Path>, callback: Option<extern "C" fn(f32)>) -> Result<Self> {
        Self::new_with_stats(model, callback, None)
    }

    fn new_with_stats(
        model: impl AsRef<Path>,
        callback: Option<extern "C" fn(f32)>,
        stats: Option<Arc<Mutex<TorchTensorStats>>>,
    ) -> Result<Self> {
        let storage = Arc::new(TorchStorage::new(model.as_ref())?);
        let torch = TorchTensors::new_from_file(model)?;
        let total = torch.0.len();
        let mut order = Vec::with_capacity(total);
        let mut tensors = HashMap::with_capacity(total);

        for tensor in torch {
            let name = rename_tensor_name(tensor.name, RENAME);
            let transpose = needs_transpose(&name, TRANSPOSE);
            let source_shape = tensor.shape;
            let shape = output_shape(&source_shape, transpose);
            let size: usize = source_shape.iter().product();
            let bytes = size * tensor.tensor_type.size();
            let start = tensor.absolute_offset as usize;
            let end = start + bytes;

            let replaced = tensors.insert(
                name.clone(),
                TorchTensorMeta {
                    source_type: tensor.tensor_type,
                    start,
                    end,
                    source_shape,
                    shape,
                    transpose,
                },
            );
            anyhow::ensure!(
                replaced.is_none(),
                "duplicate tensor name after renaming: {name}"
            );
            order.push(name);
        }

        let progress = callback.map(|callback| TorchLoadProgress {
            callback,
            total: total.max(1),
            seen: Mutex::new(HashSet::with_capacity(total)),
        });

        Ok(Self {
            storage,
            order,
            tensors,
            progress,
            stats,
        })
    }

    fn views(&self) -> Vec<(String, TorchView)> {
        self.order
            .iter()
            .map(|name| {
                let tensor = self
                    .tensors
                    .get(name)
                    .expect("tensor metadata should exist")
                    .clone();
                let view = TorchView {
                    storage: self.storage.clone(),
                    tensor,
                };
                (name.clone(), view)
            })
            .collect()
    }
}

impl Reader for TorchReader {
    fn names(&self) -> Vec<&str> {
        self.order.iter().map(|name| name.as_str()).collect()
    }

    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>, safetensors::SafeTensorError> {
        self.tensors
            .get(name)
            .map(|tensor| tensor.shape.clone())
            .ok_or(safetensors::SafeTensorError::TensorNotFound(
                name.to_string(),
            ))
    }

    fn tensor(
        &self,
        name: &str,
    ) -> Result<(Dtype, Vec<usize>, Cow<'_, [u8]>), safetensors::SafeTensorError> {
        let start = std::time::Instant::now();
        let tensor = self
            .tensors
            .get(name)
            .ok_or(safetensors::SafeTensorError::TensorNotFound(
                name.to_string(),
            ))?;
        if let Some(progress) = &self.progress {
            progress.report(name);
        }
        let data = tensor.load(&self.storage)?;
        if let Some(stats) = &self.stats {
            let mut stats = stats.lock().expect("torch tensor stats mutex poisoned");
            stats.calls += 1;
            stats.bytes += tensor.data_len();
            stats.elapsed += start.elapsed();
            match (&tensor.source_type, tensor.transpose) {
                (TensorType::Float16, false) => {
                    stats.direct_calls += 1;
                    stats.direct_bytes += tensor.data_len();
                }
                (_, true) => {
                    stats.transpose_calls += 1;
                    stats.transpose_bytes += tensor.data_len();
                }
                _ => {
                    stats.convert_calls += 1;
                    stats.convert_bytes += tensor.data_len();
                    match tensor.source_type {
                        TensorType::Float32 => {
                            stats.convert_f32_calls += 1;
                            stats.convert_f32_bytes += tensor.data_len();
                        }
                        TensorType::BFloat16 => {
                            stats.convert_bf16_calls += 1;
                            stats.convert_bf16_bytes += tensor.data_len();
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok((Dtype::F16, tensor.shape.clone(), data))
    }
}

struct TorchView {
    storage: Arc<TorchStorage>,
    tensor: TorchTensorMeta,
}

impl View for TorchView {
    fn dtype(&self) -> Dtype {
        Dtype::F16
    }

    fn shape(&self) -> &[usize] {
        &self.tensor.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        self.tensor
            .load(&self.storage)
            .expect("torch tensor view should be valid")
    }

    fn data_len(&self) -> usize {
        self.tensor.data_len()
    }
}

fn rename_tensor_name(
    name: String,
    rename: impl IntoIterator<Item = (&'static str, &'static str)>,
) -> String {
    rename
        .into_iter()
        .fold(name, |name, (from, to)| name.replace(from, to))
}

fn needs_transpose(name: &str, transpose: impl IntoIterator<Item = &'static str>) -> bool {
    transpose.into_iter().any(|pattern| name.contains(pattern))
}

fn output_shape(shape: &[usize], transpose: bool) -> Vec<usize> {
    let mut shape = shape.to_vec();
    if transpose {
        let len = shape.len();
        assert!(len >= 2, "transposed tensor should be at least 2d");
        shape.swap(len - 1, len - 2);
    }
    shape
}

fn transpose_last_two(
    shape: &[usize],
    mut value_at: impl FnMut(usize) -> f16,
) -> Result<Vec<f16>, safetensors::SafeTensorError> {
    if shape.len() < 2 {
        return Err(safetensors::SafeTensorError::TensorInvalidInfo);
    }

    let len = shape.iter().product::<usize>();
    let num_col = shape[shape.len() - 1];
    let num_row = shape[shape.len() - 2];
    let num_batch = shape[..shape.len() - 2].iter().product::<usize>().max(1);
    let mut transposed = vec![f16::ZERO; len];

    for batch in 0..num_batch {
        let offset = batch * num_col * num_row;
        for row in 0..num_row {
            for col in 0..num_col {
                let from = offset + row * num_col + col;
                let to = offset + col * num_row + row;
                transposed[to] = value_at(from);
            }
        }
    }

    Ok(transposed)
}

const PARALLEL_CONVERT_MIN_ELEMENTS: usize = 4 * 1024 * 1024;
const MAX_PARALLEL_CONVERT_THREADS: usize = 16;
const CONVERT_THREAD_STACK_SIZE: usize = 256 * 1024;

fn parallel_workers(len: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    if available <= 1 || len < PARALLEL_CONVERT_MIN_ELEMENTS {
        return 1;
    }
    let chunks = len.div_ceil(PARALLEL_CONVERT_MIN_ELEMENTS);
    available
        .min(MAX_PARALLEL_CONVERT_THREADS)
        .min(chunks)
        .max(1)
}

fn convert_thread_pool() -> &'static ThreadPool {
    CONVERT_THREAD_POOL.get_or_init(|| {
        let workers = std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1)
            .min(MAX_PARALLEL_CONVERT_THREADS)
            .max(1);
        ThreadPoolBuilder::new()
            .num_threads(workers)
            .stack_size(CONVERT_THREAD_STACK_SIZE)
            .thread_name(|index| format!("web-rwkv-convert-{index}"))
            .build()
            .expect("conversion thread pool")
    })
}

fn parallel_convert_bf16_to_f16(src: &[bf16]) -> Vec<f16> {
    let workers = parallel_workers(src.len());
    if workers == 1 {
        let mut out = uninit_f16_buffer(src.len());
        convert_bf16_to_f16_into(&mut out, src);
        return unsafe { assume_init_f16_buffer(out) };
    }

    let chunk_len = src.len().div_ceil(workers);
    let mut out = uninit_f16_buffer(src.len());
    convert_thread_pool().install(|| {
        out.par_chunks_mut(chunk_len)
            .zip(src.par_chunks(chunk_len))
            .for_each(|(dst, src)| {
                convert_bf16_to_f16_into(dst, src);
            });
    });
    unsafe { assume_init_f16_buffer(out) }
}

fn parallel_convert_f32_to_f16(src: &[f32]) -> Vec<f16> {
    let workers = parallel_workers(src.len());
    if workers == 1 {
        let mut out = vec![f16::ZERO; src.len()];
        out.convert_from_f32_slice(src);
        return out;
    }

    let chunk_len = src.len().div_ceil(workers);
    let mut out = vec![f16::ZERO; src.len()];
    convert_thread_pool().install(|| {
        out.par_chunks_mut(chunk_len)
            .zip(src.par_chunks(chunk_len))
            .for_each(|(dst, src)| {
                dst.convert_from_f32_slice(src);
            });
    });
    out
}

fn f16_vec_into_bytes(data: Vec<f16>) -> Vec<u8> {
    let mut data = ManuallyDrop::new(data);
    let len = data.len() * size_of::<f16>();
    let cap = data.capacity() * size_of::<f16>();
    let ptr = data.as_mut_ptr() as *mut u8;
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

fn uninit_f16_buffer(len: usize) -> Vec<MaybeUninit<f16>> {
    let mut out = Vec::with_capacity(len);
    unsafe {
        // `MaybeUninit<f16>` may hold uninitialized elements until each slot is written.
        out.set_len(len);
    }
    out
}

unsafe fn assume_init_f16_buffer(data: Vec<MaybeUninit<f16>>) -> Vec<f16> {
    let mut data = ManuallyDrop::new(data);
    // All call sites fully initialize every element before reinterpreting the buffer as `Vec<f16>`.
    unsafe { Vec::from_raw_parts(data.as_mut_ptr().cast::<f16>(), data.len(), data.capacity()) }
}

fn convert_bf16_to_f16_into(dst: &mut [MaybeUninit<f16>], src: &[bf16]) {
    debug_assert_eq!(dst.len(), src.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            unsafe {
                convert_bf16_to_f16_into_x86_avx2(dst, src);
            }
            return;
        }
    }

    convert_bf16_to_f16_into_scalar(dst, src);
}

fn convert_bf16_to_f16_into_scalar(dst: &mut [MaybeUninit<f16>], src: &[bf16]) {
    for (dst, src) in dst.iter_mut().zip(src.iter()) {
        dst.write(f16::from_f32(src.to_f32()));
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,f16c")]
unsafe fn convert_bf16_to_f16_into_x86_avx2(dst: &mut [MaybeUninit<f16>], src: &[bf16]) {
    debug_assert_eq!(dst.len(), src.len());

    let src_bits: &[u16] = bytemuck::cast_slice(src);
    let mut offset = 0;

    while offset + 8 <= src_bits.len() {
        let src_ptr = unsafe { src_bits.as_ptr().add(offset) } as *const __m128i;
        let dst_ptr = unsafe { dst.as_mut_ptr().add(offset) } as *mut __m128i;
        let packed = unsafe { _mm_loadu_si128(src_ptr) };
        let widened = _mm256_cvtepu16_epi32(packed);
        let shifted = _mm256_slli_epi32(widened, 16);
        let values = _mm256_castsi256_ps(shifted);
        let halves = _mm256_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT);
        unsafe { _mm_storeu_si128(dst_ptr, halves) };
        offset += 8;
    }

    convert_bf16_to_f16_into_scalar(&mut dst[offset..], &src[offset..]);
}

pub const RENAME: [(&str, &str); 4] = [
    ("time_faaaa", "time_first"),
    ("time_maa", "time_mix"),
    ("lora_A", "lora.0"),
    ("lora_B", "lora.1"),
];

pub const TRANSPOSE: [&str; 14] = [
    "time_mix_w1",
    "time_mix_w2",
    "time_decay_w1",
    "time_decay_w2",
    "w1",
    "w2",
    "a1",
    "a2",
    "g1",
    "g2",
    "v1",
    "v2",
    "time_state",
    "lora.0",
];

fn load_runtime_pth(
    model: impl AsRef<Path>,
    quant: usize,
    quant_nf4: usize,
    quant_sf4: usize,
    rescale: Option<usize>,
    extended: bool,
    fp16: bool,
    batch: usize,
    callback: Option<extern "C" fn(f32)>,
) -> Result<WktvRuntime> {
    let tokio = Arc::new(tokio::runtime::Runtime::new()?);
    let _tokio = tokio.clone();

    _tokio.block_on(async move {
        let runtime = {
            let model = TorchReader::new(&model, callback)?;
            let info = Loader::info(&model)?;
            log::info!("{:#?}", info);

            let context = create_context(&info).await?;
            log::info!("{:#?}", context.adapter.get_info());

            let quant = (0..quant)
                .map(|layer| (layer, Quant::Int8))
                .chain((0..quant_nf4).map(|layer| (layer, Quant::NF4)))
                .chain((0..quant_sf4).map(|layer| (layer, Quant::SF4)))
                .collect();

            let builder = ModelBuilder::new(&context, model).quant(quant);
            let builder = match rescale {
                Some(rescale) => builder.rescale(rescale),
                None => builder,
            };
            match info.version {
                ModelVersion::V4 => {
                    if fp16 {
                        let model = builder.build_v4().await?;
                        let bundle = v4::Bundle::<f16>::new(model, batch);
                        let state = Arc::new(bundle.state());
                        let runtime = TokioRuntime::new(bundle).await;
                        WktvRuntime {
                            runtime,
                            info,
                            state,
                            context,
                            tokio,
                        }
                    } else {
                        let model = builder.build_v4().await?;
                        let bundle = v4::Bundle::<f32>::new(model, batch);
                        let state = Arc::new(bundle.state());
                        let runtime = TokioRuntime::new(bundle).await;
                        WktvRuntime {
                            runtime,
                            info,
                            state,
                            context,
                            tokio,
                        }
                    }
                }
                ModelVersion::V5 => {
                    if fp16 {
                        let model = builder.build_v5().await?;
                        let bundle = v5::Bundle::<f16>::new(model, batch);
                        let state = Arc::new(bundle.state());
                        let runtime = TokioRuntime::new(bundle).await;
                        WktvRuntime {
                            runtime,
                            info,
                            state,
                            context,
                            tokio,
                        }
                    } else {
                        let model = builder.build_v5().await?;
                        let bundle = v5::Bundle::<f32>::new(model, batch);
                        let state = Arc::new(bundle.state());
                        let runtime = TokioRuntime::new(bundle).await;
                        WktvRuntime {
                            runtime,
                            info,
                            state,
                            context,
                            tokio,
                        }
                    }
                }
                ModelVersion::V6 => {
                    if fp16 {
                        let model = builder.build_v6().await?;
                        let bundle = match extended {
                            true => {
                                let hooks = make_hooks_extended_v6(&info)?;
                                v6::Bundle::<f16>::new_with_hooks(model, batch, hooks)
                            }
                            false => v6::Bundle::<f16>::new(model, batch),
                        };
                        let state = Arc::new(bundle.state());
                        let runtime = TokioRuntime::new(bundle).await;
                        WktvRuntime {
                            runtime,
                            info,
                            state,
                            context,
                            tokio,
                        }
                    } else {
                        let model = builder.build_v6().await?;
                        let bundle = match extended {
                            true => {
                                let hooks = make_hooks_extended_v6(&info)?;
                                v6::Bundle::<f32>::new_with_hooks(model, batch, hooks)
                            }
                            false => v6::Bundle::<f32>::new(model, batch),
                        };
                        let state = Arc::new(bundle.state());
                        let runtime = TokioRuntime::new(bundle).await;
                        WktvRuntime {
                            runtime,
                            info,
                            state,
                            context,
                            tokio,
                        }
                    }
                }
                ModelVersion::V7 => {
                    if fp16 {
                        let model = builder.build_v7().await?;
                        let bundle = match extended {
                            true => {
                                let hooks = make_hooks_extended_v7(&info)?;
                                v7::Bundle::<f16>::new_with_hooks(model, batch, hooks)
                            }
                            false => v7::Bundle::<f16>::new(model, batch),
                        };
                        let state = Arc::new(bundle.state());
                        let runtime = TokioRuntime::new(bundle).await;
                        WktvRuntime {
                            runtime,
                            info,
                            state,
                            context,
                            tokio,
                        }
                    } else {
                        let model = builder.build_v7().await?;
                        let bundle = match extended {
                            true => {
                                let hooks = make_hooks_extended_v7(&info)?;
                                v7::Bundle::<f32>::new_with_hooks(model, batch, hooks)
                            }
                            false => v7::Bundle::<f32>::new(model, batch),
                        };
                        let state = Arc::new(bundle.state());
                        let runtime = TokioRuntime::new(bundle).await;
                        WktvRuntime {
                            runtime,
                            info,
                            state,
                            context,
                            tokio,
                        }
                    }
                }
            }
        };
        trim_process_working_set();
        if let Some(cb) = callback {
            cb(1.0);
        }
        Ok(runtime)
    })
}

/// Initialize logger and RNG. Call this once before everything.
#[no_mangle]
pub extern "C" fn init(seed: u64) {
    let _ = simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("web_rwkv_ffi", log::LevelFilter::Info)
        .init();
    fastrand::seed(seed);
}

/// Set the RNG seed.
#[no_mangle]
pub extern "C" fn seed(seed: u64) {
    fastrand::seed(seed);
}

/// Load a runtime.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
#[no_mangle]
pub unsafe extern "C" fn load(
    model: *const c_char,
    quant: usize,
    quant_nf4: usize,
    quant_sf4: usize,
    fp16: bool,
    batch: usize,
) -> i32 {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime(model, quant, quant_nf4, quant_sf4, None, false, fp16, batch) {
        Ok(runtime) => {
            let mut rt = RUNTIME.write().unwrap();
            rt.replace(runtime);
            return 0;
        }
        Err(err) => {
            log::error!("{err}");
            return -1;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn release() {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return;
        };
        runtime
    };
    let _ = runtime.runtime;
    let _ = runtime.context;
    let _ = runtime.tokio;
    let _ = runtime.state;
    let mut rt = RUNTIME.write().unwrap();
    rt.take();
}

/// Load a runtime from prefab.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
#[no_mangle]
pub unsafe extern "C" fn load_prefab(model: *const c_char, fp16: bool, batch: usize) -> i32 {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime_prefab(model, fp16, batch) {
        Ok(runtime) => {
            let mut rt = RUNTIME.write().unwrap();
            rt.replace(runtime);
            return 0;
        }
        Err(err) => {
            log::error!("{err}");
            return -1;
        }
    }
}

/// Load a runtime from pth.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
#[no_mangle]
pub unsafe extern "C" fn load_pth(
    model: *const c_char,
    quant: usize,
    quant_nf4: usize,
    quant_sf4: usize,
    fp16: bool,
    batch: usize,
    callback: Option<extern "C" fn(f32)>,
) -> i32 {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime_pth(
        model, quant, quant_nf4, quant_sf4, None, false, fp16, batch, callback,
    ) {
        Ok(runtime) => {
            let mut rt = RUNTIME.write().unwrap();
            rt.replace(runtime);
            return 0;
        }
        Err(err) => {
            log::error!("{err}");
            return -1;
        }
    }
}

pub fn convert_safetensors(input: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<()> {
    let tokio = Arc::new(tokio::runtime::Runtime::new()?);
    let _tokio = tokio.clone();

    _tokio.block_on(async move {
        let reader = TorchReader::new(&input, None)?;
        let data = reader.views();
        safetensors::serialize_to_file(data, None, output.as_ref())?;
        Ok(())
    })
}

/// Convert a pth file to a st file.
///
/// # Safety
///
/// The caller must ensure that `input_path` and `output_path` are valid.
#[no_mangle]
pub unsafe extern "C" fn convert_pth_to_st(
    input_path: *const c_char,
    output_path: *const c_char,
) -> i32 {
    let input_path = unsafe { CStr::from_ptr(input_path).to_string_lossy().to_string() };
    let output_path = unsafe { CStr::from_ptr(output_path).to_string_lossy().to_string() };

    let ret = match convert_safetensors(input_path, output_path) {
        Ok(_) => 0,
        Err(err) => {
            log::error!("{err}");
            -1
        }
    };
    ret
}

/// Load a runtime with `rescale` layers specified.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
#[no_mangle]
pub unsafe extern "C" fn load_with_rescale(
    model: *const c_char,
    quant: usize,
    quant_nf4: usize,
    quant_sf4: usize,
    rescale: usize,
    fp16: bool,
    batch: usize,
) -> i32 {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime(
        model,
        quant,
        quant_nf4,
        quant_sf4,
        Some(rescale),
        false,
        fp16,
        batch,
    ) {
        Ok(runtime) => {
            let mut rt = RUNTIME.write().unwrap();
            rt.replace(runtime);
            return 0;
        }
        Err(err) => {
            log::error!("{err}");
            return -1;
        }
    }
}

/// Load a runtime with extended hooks.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
#[no_mangle]
pub unsafe extern "C" fn load_extended(
    model: *const c_char,
    quant: usize,
    quant_nf4: usize,
    quant_sf4: usize,
    fp16: bool,
    batch: usize,
) -> i32 {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime(model, quant, quant_nf4, quant_sf4, None, true, fp16, batch) {
        Ok(runtime) => {
            let mut rt = RUNTIME.write().unwrap();
            rt.replace(runtime);
            return 0;
        }
        Err(err) => {
            log::error!("{err}");
            return -1;
        }
    }
}

/// Clear the model state.
#[no_mangle]
pub extern "C" fn clear_state(batch: usize) {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return;
        };
        runtime
    };
    let tensor = runtime.state.init();
    let _ = runtime.state.load(tensor, batch);
}

/// Generate the next token prediction given the input tokens and a sampler.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
#[no_mangle]
pub unsafe extern "C" fn infer(tokens: *const u32, len: usize, sampler: Sampler) -> u32 {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return 0;
        };
        runtime
    };

    let tokens: Vec<Token> = unsafe { std::slice::from_raw_parts(tokens, len) }
        .iter()
        .map(|t| Token::Token(*t))
        .collect();
    if tokens.is_empty() {
        log::error!("input cannot be empty");
        return 0;
    }

    let tokio = runtime.tokio.clone();
    tokio.block_on(async move {
        let context = &runtime.context;
        let mut inference = Some(RnnInput::new(
            vec![RnnInputBatch {
                tokens,
                option: RnnOption::Last,
            }],
            128,
        ));
        let output = loop {
            let input = inference.take().unwrap();
            let (input, output) = match runtime.runtime.infer(input).await {
                Ok(result) => result,
                Err(err) => {
                    log::error!("Inference error: {err}");
                    return 0;
                }
            };
            let output = output[0].0.clone();

            if input.batches[0].tokens.is_empty() {
                if sampler.top_k > 1 {
                    let output = softmax_one(context, output).await.expect("softmax failed");
                    break output.to_vec();
                } else {
                    break output.to_vec();
                }
            }
            inference.replace(input);
        };
        if sampler.top_k > 1 {
            sampler.sample(&output)
        } else {
            output
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.total_cmp(y))
                .unwrap()
                .0 as u32
        }
    })
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelOutput {
    pub len: usize,
    pub data: *mut f32,
}

impl ModelOutput {
    pub fn empty() -> ModelOutput {
        ModelOutput::from(vec![])
    }
}

impl From<Vec<f32>> for ModelOutput {
    fn from(value: Vec<f32>) -> Self {
        let mut value = std::mem::ManuallyDrop::new(value);
        let len = value.len();
        let data = value.as_mut_ptr();
        ModelOutput { data, len }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelOutputBatch {
    pub batch: usize,
    pub len: usize,
    pub data: *mut f32,
}

impl ModelOutputBatch {
    pub fn empty() -> ModelOutputBatch {
        ModelOutputBatch {
            batch: 0,
            len: 0,
            data: std::ptr::null_mut(),
        }
    }
}

impl From<Vec<Vec<f32>>> for ModelOutputBatch {
    fn from(value: Vec<Vec<f32>>) -> Self {
        let batch = value.len();
        let len = value[0].len();
        let mut data =
            std::mem::ManuallyDrop::new(value.into_iter().flat_map(|v| v).collect::<Vec<_>>());
        let data = data.as_mut_ptr();
        ModelOutputBatch { batch, len, data }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateRaw {
    pub len: usize,
    pub data: *mut f32,
}

impl StateRaw {
    pub fn empty() -> StateRaw {
        StateRaw::from(vec![])
    }
}

impl From<Vec<f32>> for StateRaw {
    fn from(value: Vec<f32>) -> Self {
        let mut value = std::mem::ManuallyDrop::new(value);
        let len = value.len();
        let data = value.as_mut_ptr();
        StateRaw { data, len }
    }
}

/// Get the model state.
#[no_mangle]
pub extern "C" fn get_state(batch: usize) -> StateRaw {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return StateRaw::empty();
        };
        runtime
    };
    let tokio = runtime.tokio.clone();
    let tensor = tokio
        .block_on(async move {
            runtime
                .state
                .back(batch)
                .await
                .map_err(|err| log::error!("{err}"))
        })
        .unwrap();
    tensor.to_vec().into()
}

/// Free the returned state vector created by the get_state function.
#[no_mangle]
pub extern "C" fn free_state(state: StateRaw) {
    let x = unsafe { std::slice::from_raw_parts_mut(state.data, state.len) };
    let x = x.as_mut_ptr();
    let _ = unsafe { Box::from_raw(x) };
}

/// Set the model state.
#[no_mangle]
pub extern "C" fn set_state(data: StateRaw, batch: usize) {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return;
        };
        runtime
    };
    let tokio = runtime.tokio.clone();
    tokio.block_on(async move {
        let shape = runtime.state.init_shape();
        let state = unsafe { std::slice::from_raw_parts(data.data, data.len) };
        let state: web_rwkv::tensor::Tensor<web_rwkv::tensor::Cpu<f32>, f32> = runtime
            .context
            .tensor_from_data(shape, state.to_vec())
            .unwrap();
        let _ = runtime.state.load(state, batch);
    });
}

/// Delete the model output vector created by the infer functions.
#[no_mangle]
pub extern "C" fn free_raw(output: ModelOutput) {
    let x = unsafe { std::slice::from_raw_parts_mut(output.data, output.len) };
    let x = x.as_mut_ptr();
    let _ = unsafe { Box::from_raw(x) };
}

#[no_mangle]
pub extern "C" fn free_raw_batch(output: ModelOutputBatch) {
    let x = unsafe { std::slice::from_raw_parts_mut(output.data, output.len * output.batch) };
    let x = x.as_mut_ptr();
    let _ = unsafe { Box::from_raw(x) };
}

/// Compute the model's raw output (next token prediction only) given the input tokens.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
#[no_mangle]
pub unsafe extern "C" fn infer_raw_last(tokens: *const u32, len: usize) -> ModelOutput {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return ModelOutput::empty();
        };
        runtime
    };

    let tokens: Vec<Token> = unsafe { std::slice::from_raw_parts(tokens, len) }
        .iter()
        .map(|t| Token::Token(*t))
        .collect();
    if tokens.is_empty() {
        log::error!("input cannot be empty");
        return ModelOutput::empty();
    }

    let tokio = runtime.tokio.clone();
    let output = tokio.block_on(async move {
        let mut inference = Some(RnnInput::new(
            vec![RnnInputBatch {
                tokens: tokens.to_vec(),
                option: RnnOption::Last,
            }],
            128,
        ));
        loop {
            let input = inference.take().unwrap();
            let (input, output) = match runtime.runtime.infer(input).await {
                Ok(result) => result,
                Err(err) => {
                    log::error!("Inference error: {err}");
                    break vec![];
                }
            };
            let output = output[0].0.clone();

            if input.batches[0].tokens.is_empty() {
                break output.to_vec();
            }
            inference.replace(input);
        }
    });

    output.into()
}

/// Compute the model's raw output (next token prediction only) given the input tokens.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` and `batch` does not exceed the actual length of `tokens`.
#[no_mangle]
pub unsafe extern "C" fn infer_raw_last_batch(
    tokens: *const *const u32,
    len: *const usize,
    batch: usize,
) -> ModelOutputBatch {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return ModelOutputBatch::empty();
        };
        runtime
    };

    let per_batch_len = unsafe { std::slice::from_raw_parts(len, batch) };
    let tokens_ptr = unsafe { std::slice::from_raw_parts(tokens, batch) };
    let mut tokens_vec = Vec::new();
    for i in 0..batch {
        let batch_tokens = unsafe { std::slice::from_raw_parts(tokens_ptr[i], per_batch_len[i]) }
            .iter()
            .map(|t| Token::Token(*t))
            .collect();
        tokens_vec.push(batch_tokens);
    }
    if tokens_vec.is_empty() {
        log::error!("input cannot be empty");
        return ModelOutputBatch::empty();
    }

    let tokio = runtime.tokio.clone();
    let output = tokio.block_on(async move {
        let mut inference = Some(RnnInput::new(
            tokens_vec
                .into_iter()
                .map(|t| RnnInputBatch::new(t, RnnOption::Last))
                .collect(),
            128,
        ));
        loop {
            let input = inference.take().unwrap();
            let (input, output) = match runtime.runtime.infer(input).await {
                Ok(result) => result,
                Err(err) => {
                    log::error!("Inference error: {err}");
                    break vec![];
                }
            };

            if input.batches.iter().all(|batch| batch.tokens.is_empty()) {
                let output = output
                    .iter()
                    .map(|batch| batch.0.clone().to_vec())
                    .collect_vec();
                break output.into();
            }
            inference.replace(input);
        }
    });

    output.into()
}

/// Compute the model's raw output (predictions of all tokens) given the input tokens.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
#[no_mangle]
pub unsafe extern "C" fn infer_raw_full(tokens: *const u32, len: usize) -> ModelOutput {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return ModelOutput::empty();
        };
        runtime
    };

    let tokens: Vec<Token> = unsafe { std::slice::from_raw_parts(tokens, len) }
        .iter()
        .map(|t| Token::Token(*t))
        .collect();
    if tokens.is_empty() {
        log::error!("input cannot be empty");
        return ModelOutput::empty();
    }

    let tokio = runtime.tokio.clone();
    let output = tokio.block_on(async move {
        let mut inference = Some(RnnInput::new(
            vec![RnnInputBatch {
                tokens: tokens.to_vec(),
                option: RnnOption::Full,
            }],
            128,
        ));
        let mut outputs = vec![];
        loop {
            let input = inference.take().unwrap();
            let (input, output) = match runtime.runtime.infer(input).await {
                Ok(result) => result,
                Err(err) => {
                    log::error!("Inference error: {err}");
                    break;
                }
            };
            let mut output = output[0].0.clone().to_vec();
            outputs.append(&mut output);

            if input.batches[0].tokens.is_empty() {
                break;
            }
            inference.replace(input);
        }
        outputs
    });

    output.into()
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Sampler {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temp: 1.0,
            top_p: 0.5,
            top_k: 128,
        }
    }
}

impl Sampler {
    pub fn sample(&self, probs: &[f32]) -> u32 {
        let sorted: Vec<_> = probs
            .iter()
            .copied()
            .enumerate()
            .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(y).reverse())
            .take(self.top_k.max(1))
            .scan((0, 0.0, 0.0), |(_, cum, _), (id, x)| {
                if *cum > self.top_p {
                    None
                } else {
                    *cum += x;
                    Some((id, *cum, x))
                }
            })
            .map(|(id, _, x)| (id, x.powf(1.0 / self.temp)))
            .collect();

        let sum: f32 = sorted.iter().map(|(_, x)| x).sum();
        let sorted: Vec<_> = sorted
            .into_iter()
            .map(|(id, x)| (id, x / sum))
            .scan((0, 0.0), |(_, cum), (id, x)| {
                *cum += x;
                Some((id, *cum))
            })
            .collect();

        let rand = fastrand::f32();
        let token = sorted
            .into_iter()
            .find_or_first(|&(_, cum)| rand <= cum)
            .map(|(id, _)| id)
            .unwrap_or_default();
        token as u32
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ModelInfoOutput {
    pub version: usize,
    pub num_layer: usize,
    pub num_hidden: usize,
    pub num_emb: usize,
    pub num_vocab: usize,
    pub num_head: usize,
}

impl Default for ModelInfoOutput {
    fn default() -> Self {
        Self {
            version: 0,
            num_layer: 0,
            num_hidden: 0,
            num_emb: 0,
            num_vocab: 0,
            num_head: 0,
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn get_model_info() -> ModelInfoOutput {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return ModelInfoOutput::default();
        };
        runtime
    };

    let info = runtime.info;
    ModelInfoOutput {
        version: match info.version {
            ModelVersion::V4 => 4,
            ModelVersion::V5 => 5,
            ModelVersion::V6 => 6,
            ModelVersion::V7 => 7,
        },
        num_layer: info.num_layer,
        num_hidden: info.num_hidden,
        num_emb: info.num_emb,
        num_vocab: info.num_vocab,
        num_head: info.num_head,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, ffi::CString, time::Instant};

    fn env_usize(name: &str, default: usize) -> usize {
        env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn env_bool(name: &str, default: bool) -> bool {
        env::var(name)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(default)
    }

    #[test]
    fn bf16_to_f16_matches_scalar_for_all_bit_patterns() {
        let src: Vec<bf16> = (u16::MIN..=u16::MAX).map(bf16::from_bits).collect();
        let expected: Vec<u16> = src
            .iter()
            .map(|value| f16::from_f32(value.to_f32()).to_bits())
            .collect();
        let mut actual = uninit_f16_buffer(src.len());
        convert_bf16_to_f16_into(&mut actual, &src);
        let actual = unsafe { assume_init_f16_buffer(actual) };
        let actual: Vec<u16> = actual.into_iter().map(f16::to_bits).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    #[ignore = "manual benchmark for load_pth wall-clock time"]
    fn bench_load_pth() {
        let model = env::var("WEB_RWKV_BENCH_MODEL")
            .unwrap_or_else(|_| "/models/rwkv7-g1f-13.3b-20260415-ctx8192.pth".to_string());
        let quant = env_usize("WEB_RWKV_BENCH_QUANT", 0);
        let quant_nf4 = env_usize("WEB_RWKV_BENCH_QUANT_NF4", 0);
        let quant_sf4 = env_usize("WEB_RWKV_BENCH_QUANT_SF4", 0);
        let fp16 = env_bool("WEB_RWKV_BENCH_FP16", true);
        let batch = env_usize("WEB_RWKV_BENCH_BATCH", 1);
        let model =
            CString::new(model.clone()).expect("benchmark model path should not contain NUL");

        init(0);

        let start = Instant::now();
        let ret = unsafe {
            load_pth(
                model.as_ptr(),
                quant,
                quant_nf4,
                quant_sf4,
                fp16,
                batch,
                None,
            )
        };
        let elapsed = start.elapsed();

        assert_eq!(ret, 0, "load_pth failed for {model:?}");
        eprintln!(
            "load_pth elapsed={:.3}s quant={quant} quant_nf4={quant_nf4} quant_sf4={quant_sf4} fp16={fp16} batch={batch}",
            elapsed.as_secs_f64(),
        );

        unsafe { release() };
    }

    #[test]
    #[ignore = "manual benchmark for safetensors load wall-clock time"]
    fn bench_load_st() {
        let model = env::var("WEB_RWKV_BENCH_MODEL").unwrap_or_else(|_| {
            "/home/molly/dist/assets/models/rwkv7-g1e-7.2b-20260301-ctx8192.st".to_string()
        });
        let quant = env_usize("WEB_RWKV_BENCH_QUANT", 0);
        let quant_nf4 = env_usize("WEB_RWKV_BENCH_QUANT_NF4", 0);
        let quant_sf4 = env_usize("WEB_RWKV_BENCH_QUANT_SF4", 0);
        let fp16 = env_bool("WEB_RWKV_BENCH_FP16", true);
        let batch = env_usize("WEB_RWKV_BENCH_BATCH", 1);
        let model =
            CString::new(model.clone()).expect("benchmark model path should not contain NUL");

        init(0);

        let start = Instant::now();
        let ret = unsafe { load(model.as_ptr(), quant, quant_nf4, quant_sf4, fp16, batch) };
        let elapsed = start.elapsed();

        assert_eq!(ret, 0, "load failed for {model:?}");
        eprintln!(
            "load_st elapsed={:.3}s quant={quant} quant_nf4={quant_nf4} quant_sf4={quant_sf4} fp16={fp16} batch={batch}",
            elapsed.as_secs_f64(),
        );

        unsafe { release() };
    }

    async fn bench_build_from_reader<R: Reader>(
        context: &Context,
        model: R,
        info: ModelInfo,
        quant: usize,
        quant_nf4: usize,
        quant_sf4: usize,
        fp16: bool,
        batch: usize,
    ) {
        let quant = (0..quant)
            .map(|layer| (layer, Quant::Int8))
            .chain((0..quant_nf4).map(|layer| (layer, Quant::NF4)))
            .chain((0..quant_sf4).map(|layer| (layer, Quant::SF4)))
            .collect();
        let builder = ModelBuilder::new(context, model).quant(quant);

        match info.version {
            ModelVersion::V4 => {
                let start = Instant::now();
                if fp16 {
                    let model = builder.build_v4().await.expect("build_v4");
                    eprintln!("phase build_v4 {:.3}s", start.elapsed().as_secs_f64());
                    let bundle = v4::Bundle::<f16>::new(model, batch);
                    let _state = bundle.state();
                    let start = Instant::now();
                    let _runtime: TokioRuntime<Rnn> = TokioRuntime::new(bundle).await;
                    eprintln!("phase runtime_new {:.3}s", start.elapsed().as_secs_f64());
                } else {
                    let model = builder.build_v4().await.expect("build_v4");
                    eprintln!("phase build_v4 {:.3}s", start.elapsed().as_secs_f64());
                    let bundle = v4::Bundle::<f32>::new(model, batch);
                    let _state = bundle.state();
                    let start = Instant::now();
                    let _runtime: TokioRuntime<Rnn> = TokioRuntime::new(bundle).await;
                    eprintln!("phase runtime_new {:.3}s", start.elapsed().as_secs_f64());
                }
            }
            ModelVersion::V5 => {
                let start = Instant::now();
                if fp16 {
                    let model = builder.build_v5().await.expect("build_v5");
                    eprintln!("phase build_v5 {:.3}s", start.elapsed().as_secs_f64());
                    let bundle = v5::Bundle::<f16>::new(model, batch);
                    let _state = bundle.state();
                    let start = Instant::now();
                    let _runtime: TokioRuntime<Rnn> = TokioRuntime::new(bundle).await;
                    eprintln!("phase runtime_new {:.3}s", start.elapsed().as_secs_f64());
                } else {
                    let model = builder.build_v5().await.expect("build_v5");
                    eprintln!("phase build_v5 {:.3}s", start.elapsed().as_secs_f64());
                    let bundle = v5::Bundle::<f32>::new(model, batch);
                    let _state = bundle.state();
                    let start = Instant::now();
                    let _runtime: TokioRuntime<Rnn> = TokioRuntime::new(bundle).await;
                    eprintln!("phase runtime_new {:.3}s", start.elapsed().as_secs_f64());
                }
            }
            ModelVersion::V6 => {
                let start = Instant::now();
                if fp16 {
                    let model = builder.build_v6().await.expect("build_v6");
                    eprintln!("phase build_v6 {:.3}s", start.elapsed().as_secs_f64());
                    let bundle = v6::Bundle::<f16>::new(model, batch);
                    let _state = bundle.state();
                    let start = Instant::now();
                    let _runtime: TokioRuntime<Rnn> = TokioRuntime::new(bundle).await;
                    eprintln!("phase runtime_new {:.3}s", start.elapsed().as_secs_f64());
                } else {
                    let model = builder.build_v6().await.expect("build_v6");
                    eprintln!("phase build_v6 {:.3}s", start.elapsed().as_secs_f64());
                    let bundle = v6::Bundle::<f32>::new(model, batch);
                    let _state = bundle.state();
                    let start = Instant::now();
                    let _runtime: TokioRuntime<Rnn> = TokioRuntime::new(bundle).await;
                    eprintln!("phase runtime_new {:.3}s", start.elapsed().as_secs_f64());
                }
            }
            ModelVersion::V7 => {
                let start = Instant::now();
                if fp16 {
                    let model = builder.build_v7().await.expect("build_v7");
                    eprintln!("phase build_v7 {:.3}s", start.elapsed().as_secs_f64());
                    let bundle = v7::Bundle::<f16>::new(model, batch);
                    let _state = bundle.state();
                    let start = Instant::now();
                    let _runtime: TokioRuntime<Rnn> = TokioRuntime::new(bundle).await;
                    eprintln!("phase runtime_new {:.3}s", start.elapsed().as_secs_f64());
                } else {
                    let model = builder.build_v7().await.expect("build_v7");
                    eprintln!("phase build_v7 {:.3}s", start.elapsed().as_secs_f64());
                    let bundle = v7::Bundle::<f32>::new(model, batch);
                    let _state = bundle.state();
                    let start = Instant::now();
                    let _runtime: TokioRuntime<Rnn> = TokioRuntime::new(bundle).await;
                    eprintln!("phase runtime_new {:.3}s", start.elapsed().as_secs_f64());
                }
            }
        }
    }

    #[test]
    #[ignore = "manual benchmark for load_pth phase timings"]
    fn bench_load_pth_phases() {
        let model = env::var("WEB_RWKV_BENCH_MODEL")
            .unwrap_or_else(|_| "/models/rwkv7-g1f-7.2b-20260414-ctx8192.pth".to_string());
        let quant = env_usize("WEB_RWKV_BENCH_QUANT", 0);
        let quant_nf4 = env_usize("WEB_RWKV_BENCH_QUANT_NF4", 0);
        let quant_sf4 = env_usize("WEB_RWKV_BENCH_QUANT_SF4", 0);
        let fp16 = env_bool("WEB_RWKV_BENCH_FP16", true);
        let batch = env_usize("WEB_RWKV_BENCH_BATCH", 1);

        init(0);

        let total_start = Instant::now();
        let tokio = Arc::new(tokio::runtime::Runtime::new().expect("tokio runtime"));
        let _tokio = tokio.clone();
        let stats = Arc::new(Mutex::new(TorchTensorStats::default()));

        _tokio.block_on(async move {
            let start = Instant::now();
            let model = TorchReader::new_with_stats(&model, None, Some(stats.clone()))
                .expect("torch reader");
            eprintln!(
                "phase torch_reader_new {:.3}s",
                start.elapsed().as_secs_f64()
            );

            let start = Instant::now();
            let info = Loader::info(&model).expect("loader info");
            eprintln!("phase loader_info {:.3}s", start.elapsed().as_secs_f64());

            let start = Instant::now();
            let context = create_context(&info).await.expect("context");
            eprintln!("phase create_context {:.3}s", start.elapsed().as_secs_f64());
            bench_build_from_reader(
                &context, model, info, quant, quant_nf4, quant_sf4, fp16, batch,
            )
            .await;

            let stats = stats.lock().expect("torch tensor stats mutex poisoned");
            eprintln!(
                "reader_stats calls={} bytes={} direct_calls={} direct_bytes={} convert_calls={} convert_bytes={} convert_f32_calls={} convert_f32_bytes={} convert_bf16_calls={} convert_bf16_bytes={} transpose_calls={} transpose_bytes={} reader_tensor_time={:.3}s",
                stats.calls,
                stats.bytes,
                stats.direct_calls,
                stats.direct_bytes,
                stats.convert_calls,
                stats.convert_bytes,
                stats.convert_f32_calls,
                stats.convert_f32_bytes,
                stats.convert_bf16_calls,
                stats.convert_bf16_bytes,
                stats.transpose_calls,
                stats.transpose_bytes,
                stats.elapsed.as_secs_f64(),
            );

            eprintln!("phase total {:.3}s", total_start.elapsed().as_secs_f64());
        });
    }

    #[test]
    #[ignore = "manual benchmark for safetensors load phase timings"]
    fn bench_load_st_phases() {
        let model = env::var("WEB_RWKV_BENCH_MODEL").unwrap_or_else(|_| {
            "/home/molly/dist/assets/models/rwkv7-g1e-7.2b-20260301-ctx8192.st".to_string()
        });
        let quant = env_usize("WEB_RWKV_BENCH_QUANT", 0);
        let quant_nf4 = env_usize("WEB_RWKV_BENCH_QUANT_NF4", 0);
        let quant_sf4 = env_usize("WEB_RWKV_BENCH_QUANT_SF4", 0);
        let fp16 = env_bool("WEB_RWKV_BENCH_FP16", true);
        let batch = env_usize("WEB_RWKV_BENCH_BATCH", 1);

        init(0);

        let total_start = Instant::now();
        let tokio = Arc::new(tokio::runtime::Runtime::new().expect("tokio runtime"));
        let _tokio = tokio.clone();

        _tokio.block_on(async move {
            let start = Instant::now();
            let file = File::open(&model).await.expect("open safetensors");
            let data = unsafe { Mmap::map(&file).expect("mmap safetensors") };
            eprintln!("phase open_mmap {:.3}s", start.elapsed().as_secs_f64());

            let start = Instant::now();
            let model = SafeTensors::deserialize(&data).expect("deserialize safetensors");
            eprintln!("phase deserialize_st {:.3}s", start.elapsed().as_secs_f64());

            let start = Instant::now();
            let info = Loader::info(&model).expect("loader info");
            eprintln!("phase loader_info {:.3}s", start.elapsed().as_secs_f64());

            let start = Instant::now();
            let context = create_context(&info).await.expect("context");
            eprintln!("phase create_context {:.3}s", start.elapsed().as_secs_f64());

            bench_build_from_reader(
                &context, model, info, quant, quant_nf4, quant_sf4, fp16, batch,
            )
            .await;

            eprintln!("phase total {:.3}s", total_start.elapsed().as_secs_f64());
        });
    }
}
