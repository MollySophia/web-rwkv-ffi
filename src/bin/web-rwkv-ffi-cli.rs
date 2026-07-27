use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicUsize, Ordering},
};

use anyhow::{anyhow, bail, Result};
use web_rwkv_ffi::{convert_prefab, convert_safetensors, init};

static LAST_PROGRESS_PERCENT: AtomicUsize = AtomicUsize::new(usize::MAX);

extern "C" fn print_progress(progress: f32) {
    let percent = (progress.clamp(0.0, 1.0) * 100.0).floor() as usize;
    let previous = LAST_PROGRESS_PERCENT.swap(percent, Ordering::Relaxed);
    if previous != percent {
        eprintln!("progress: {percent}%");
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    ConvertPthToSt,
    ConvertPthToPrefab,
}

#[derive(Debug, Clone)]
struct Options {
    command: Command,
    input: PathBuf,
    output: PathBuf,
    quant: usize,
    quant_nf4: usize,
    quant_sf4: usize,
    fp16: bool,
    batch: usize,
    progress: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            command: Command::ConvertPthToSt,
            input: PathBuf::new(),
            output: PathBuf::new(),
            quant: 0,
            quant_nf4: 0,
            quant_sf4: 0,
            fp16: true,
            batch: 1,
            progress: true,
        }
    }
}

impl Options {
    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self> {
        let args: Vec<String> = args
            .into_iter()
            .map(|arg| arg.into_string().map_err(|arg| anyhow!("invalid utf-8 argument: {arg:?}")))
            .collect::<Result<_>>()?;

        let Some((command, tail)) = args.split_first() else {
            print_help();
            process::exit(0);
        };

        if matches!(command.as_str(), "-h" | "--help" | "help") {
            print_help();
            process::exit(0);
        }

        let mut options = Options {
            command: match command.as_str() {
                "convert-pth-to-st" => Command::ConvertPthToSt,
                "convert-pth-to-prefab" => Command::ConvertPthToPrefab,
                other => bail!("unknown command: {other}"),
            },
            ..Default::default()
        };

        let mut index = 0;
        while index < tail.len() {
            match tail[index].as_str() {
                "--help" | "-h" => {
                    print_command_help(options.command);
                    process::exit(0);
                }
                "--input" => {
                    options.input = PathBuf::from(next_value(tail, &mut index, "--input")?);
                }
                "--output" => {
                    options.output = PathBuf::from(next_value(tail, &mut index, "--output")?);
                }
                "--quant" => {
                    options.quant = next_value(tail, &mut index, "--quant")?.parse()?;
                }
                "--quant-nf4" => {
                    options.quant_nf4 = next_value(tail, &mut index, "--quant-nf4")?.parse()?;
                }
                "--quant-sf4" => {
                    options.quant_sf4 = next_value(tail, &mut index, "--quant-sf4")?.parse()?;
                }
                "--batch" => {
                    options.batch = next_value(tail, &mut index, "--batch")?.parse()?;
                }
                "--fp16" => {
                    options.fp16 = true;
                }
                "--fp32" => {
                    options.fp16 = false;
                }
                "--no-progress" => {
                    options.progress = false;
                }
                other => bail!("unknown option: {other}"),
            }
            index += 1;
        }

        options.validate()?;
        Ok(options)
    }

    fn validate(&self) -> Result<()> {
        if self.input.as_os_str().is_empty() {
            bail!("missing required option: --input");
        }
        if self.output.as_os_str().is_empty() {
            bail!("missing required option: --output");
        }

        match self.command {
            Command::ConvertPthToSt => {
                if self.quant != 0 || self.quant_nf4 != 0 || self.quant_sf4 != 0 {
                    bail!("quantization options are not used by convert-pth-to-st");
                }
                if self.batch != 1 {
                    bail!("--batch is not used by convert-pth-to-st");
                }
                if !self.fp16 {
                    bail!("--fp32 is not used by convert-pth-to-st");
                }
            }
            Command::ConvertPthToPrefab => {}
        }

        Ok(())
    }
}

fn next_value<'a>(args: &'a [String], index: &mut usize, flag: &str) -> Result<&'a str> {
    let value_index = *index + 1;
    let Some(value) = args.get(value_index) else {
        bail!("missing value for {flag}");
    };
    *index = value_index;
    Ok(value)
}

fn print_help() {
    eprintln!(
        "Usage:\n  web-rwkv-ffi-cli <command> [options]\n\nCommands:\n  convert-pth-to-st\n  convert-pth-to-prefab\n\nRun `web-rwkv-ffi-cli <command> --help` for command-specific options."
    );
}

fn print_command_help(command: Command) {
    match command {
        Command::ConvertPthToSt => eprintln!(
            "Usage:\n  web-rwkv-ffi-cli convert-pth-to-st --input <model.pth> --output <model.st>\n"
        ),
        Command::ConvertPthToPrefab => eprintln!(
            "Usage:\n  web-rwkv-ffi-cli convert-pth-to-prefab --input <model.pth> --output <model.prefab> [--quant <layers>] [--quant-nf4 <layers>] [--quant-sf4 <layers>] [--fp16|--fp32] [--batch <n>] [--no-progress]\n"
        ),
    }
}

fn ensure_parent_exists(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let options = Options::parse(env::args_os().skip(1))?;
    ensure_parent_exists(&options.output)?;
    init(0);

    match options.command {
        Command::ConvertPthToSt => {
            convert_safetensors(&options.input, &options.output)?;
            eprintln!("saved: {}", options.output.display());
        }
        Command::ConvertPthToPrefab => {
            LAST_PROGRESS_PERCENT.store(usize::MAX, Ordering::Relaxed);
            let callback = options.progress.then_some(print_progress as extern "C" fn(f32));
            convert_prefab(
                &options.input,
                &options.output,
                options.quant,
                options.quant_nf4,
                options.quant_sf4,
                options.fp16,
                options.batch,
                callback,
            )?;
            eprintln!("saved: {}", options.output.display());
        }
    }

    Ok(())
}
