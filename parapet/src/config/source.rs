// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use super::error::ConfigError;

/// Abstraction over where config YAML comes from.
///
/// `FileSource` reads from disk; `StringSource` provides content directly
/// (used in tests to avoid file I/O).
pub trait ConfigSource {
    fn load(&self) -> Result<String, ConfigError>;
}

/// Loads config from a file on disk.
pub struct FileSource {
    pub path: PathBuf,
}

impl ConfigSource for FileSource {
    fn load(&self) -> Result<String, ConfigError> {
        Ok(std::fs::read_to_string(&self.path)?)
    }
}

/// Provides config content directly as a string. Used for testing.
pub struct StringSource {
    pub content: String,
}

impl ConfigSource for StringSource {
    fn load(&self) -> Result<String, ConfigError> {
        Ok(self.content.clone())
    }
}
