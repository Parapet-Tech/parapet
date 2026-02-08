use super::error::ConfigError;

/// Resolves `${VAR_NAME}` references in a string from environment variables.
/// Returns `ConfigError::UndefinedVariable` if a referenced variable is not set.
pub fn resolve_variables(input: &str) -> Result<String, ConfigError> {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '$' && chars.peek() == Some(&'{') {
            chars.next(); // consume '{'
            let mut var_name = String::new();
            let mut found_close = false;
            for c in chars.by_ref() {
                if c == '}' {
                    found_close = true;
                    break;
                }
                var_name.push(c);
            }
            if !found_close || var_name.is_empty() {
                // Malformed interpolation -- treat literally
                result.push('$');
                result.push('{');
                result.push_str(&var_name);
                continue;
            }
            let value = std::env::var(&var_name).map_err(|_| ConfigError::UndefinedVariable {
                name: var_name.clone(),
            })?;
            result.push_str(&value);
        } else {
            result.push(ch);
        }
    }

    Ok(result)
}
