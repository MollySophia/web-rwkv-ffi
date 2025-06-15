use std::ffi::CString;
use web_rwkv_ffi::{init, load_prefab, infer, Sampler, release};
use sysinfo::{System, SystemExt, ProcessExt};

fn main() {
    init(42);

    let mut sys = System::new_all();
    let process_id = std::process::id();

    let mut print_memory_usage = |stage: &str| {
        sys.refresh_process(sysinfo::Pid::from(process_id as usize));
        if let Some(process) = sys.process(sysinfo::Pid::from(process_id as usize)) {
            let memory_usage_mb = process.memory() as f64 / 1024.0 / 1024.0;
            println!("Memory Usage [{}]: {:.2} MB", stage, memory_usage_mb);
        }
    };

    print_memory_usage("Before Loading");
    let model_path = CString::new("/Users/molly/Downloads/dist/assets/models/rwkv7-g1-2.9b-20250519-ctx4096-nf4.prefab").unwrap();
    unsafe {
        load_prefab(model_path.as_ptr());
    }
    print_memory_usage("After Loading");
    let tokens: Vec<u16> = vec![0];

    let sampler = Sampler {
        temp: 1.0,
        top_p: 0.5,
        top_k: 128,
    };

    let next_token = unsafe {
        infer(tokens.as_ptr(), tokens.len(), sampler)
    };

    println!("next token: {}", next_token);

    unsafe {
        release();
    }
}
