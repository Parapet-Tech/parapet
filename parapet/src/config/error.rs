// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

/// All errors that can occur during config loading and validation.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config source: {0}")]
    IoError(#[from] std::io::Error),

    #[error("failed to parse YAML: {0}")]
    YamlError(#[from] serde_yaml::Error),

    #[error("validation error: {0}")]
    Validation(String),

    #[error("invalid regex pattern \"{pattern}\": {source}")]
    InvalidRegex {
        pattern: String,
        source: regex::Error,
    },

    #[error("undefined variable ${{{name}}} in config (not set in environment)")]
    UndefinedVariable { name: String },
}
