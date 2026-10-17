// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(parapet::cli::run_from_env() as u8)
}
