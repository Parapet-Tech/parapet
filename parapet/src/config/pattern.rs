use std::fmt;

use regex::Regex;

use super::error::ConfigError;

/// A pre-compiled regex pattern. Wraps `regex::Regex` with the original
/// pattern string preserved for debugging/display.
#[derive(Clone)]
pub struct CompiledPattern {
    pub pattern: String,
    pub regex: Regex,
}

impl CompiledPattern {
    /// Compile a regex pattern, returning `ConfigError::InvalidRegex` on failure.
    pub fn compile(pattern: &str) -> Result<Self, ConfigError> {
        let regex = Regex::new(pattern).map_err(|e| ConfigError::InvalidRegex {
            pattern: pattern.to_string(),
            source: e,
        })?;
        Ok(Self {
            pattern: pattern.to_string(),
            regex,
        })
    }

    /// Test whether the pattern matches the given text.
    pub fn is_match(&self, text: &str) -> bool {
        self.regex.is_match(text)
    }
}

impl fmt::Debug for CompiledPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledPattern")
            .field("pattern", &self.pattern)
            .finish()
    }
}

impl PartialEq for CompiledPattern {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
    }
}
