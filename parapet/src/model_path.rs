// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Shared model path resolution used by both the engine runtime and the
// `parapet-fetch` CLI.
//
// Resolution order:
//   1. `config_model_dir` (from YAML `model_dir` or CLI `--model-dir`)
//   2. `$PARAPET_MODEL_DIR` environment variable
//   3. `~/.parapet/models/` (platform default)

use std::path::PathBuf;

/// Resolve the directory where model files are stored.
///
/// Precedence: `config_model_dir` > `$PARAPET_MODEL_DIR` > `~/.parapet/models/`
///
/// Returns `None` only if no home directory can be determined and neither
/// of the explicit overrides are set (very unlikely in practice).
pub fn resolve_model_dir(config_model_dir: Option<&str>) -> Option<PathBuf> {
    // 1. Explicit config/CLI override.
    if let Some(dir) = config_model_dir {
        return Some(PathBuf::from(dir));
    }

    // 2. Environment variable.
    if let Ok(dir) = std::env::var("PARAPET_MODEL_DIR") {
        if !dir.is_empty() {
            return Some(PathBuf::from(dir));
        }
    }

    // 3. Platform default: ~/.parapet/models/
    dirs::home_dir().map(|h| h.join(".parapet").join("models"))
}

/// Build the full path to a specific model's directory.
///
/// Returns `resolve_model_dir() / model_name`, e.g.
/// `~/.parapet/models/pg2-86m/`
pub fn model_dir_for(config_model_dir: Option<&str>, model_name: &str) -> Option<PathBuf> {
    resolve_model_dir(config_model_dir).map(|base| base.join(model_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn explicit_config_takes_precedence() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::set_var("PARAPET_MODEL_DIR", "/env/path");
        let result = resolve_model_dir(Some("/config/path"));
        assert_eq!(result.unwrap(), PathBuf::from("/config/path"));
        std::env::remove_var("PARAPET_MODEL_DIR");
    }

    #[test]
    fn env_var_takes_precedence_over_default() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::set_var("PARAPET_MODEL_DIR", "/env/models");
        let result = resolve_model_dir(None);
        assert_eq!(result.unwrap(), PathBuf::from("/env/models"));
        std::env::remove_var("PARAPET_MODEL_DIR");
    }

    #[test]
    fn falls_back_to_home_dir() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("PARAPET_MODEL_DIR");
        let result = resolve_model_dir(None);
        // Should end with .parapet/models
        let path = result.unwrap();
        assert!(path.ends_with(".parapet/models") || path.ends_with(".parapet\\models"));
    }

    #[test]
    fn model_dir_for_appends_model_name() {
        let result = model_dir_for(Some("/models"), "pg2-86m");
        assert_eq!(result.unwrap(), PathBuf::from("/models/pg2-86m"));
    }

    #[test]
    fn empty_env_var_falls_through() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::set_var("PARAPET_MODEL_DIR", "");
        let result = resolve_model_dir(None);
        // Should fall through to home dir, not return empty PathBuf
        let path = result.unwrap();
        assert!(!path.as_os_str().is_empty());
        std::env::remove_var("PARAPET_MODEL_DIR");
    }
}
