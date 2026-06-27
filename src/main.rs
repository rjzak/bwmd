// SPDX-License-Identifier: Apache-2.0

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.is_empty() {
        println!("Usage: {} <file1> <file2>", args[0]);
        std::process::exit(1);
    }
    if args.len() != 3 {
        println!("Usage: {} <file1> <file2>", args[0]);
        std::process::exit(1);
    }
    let a = std::fs::read(&args[1]).unwrap();
    let b = std::fs::read(&args[2]).unwrap();
    println!("BWMD distance: {}", bwmd::distance(&a, &b));
}
