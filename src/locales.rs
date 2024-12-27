use serde_yaml::from_str;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LocaleError {
    #[error("Failed to load locale file: {0}")]
    LoadError(String),
    #[error("Failed to parse locale file: {0}")]
    ParseError(String),
    #[error("Locale not found: {0}")]
    LocaleNotFound(String),
}

impl From<std::io::Error> for LocaleError {
    fn from(err: std::io::Error) -> Self {
        LocaleError::LoadError(err.to_string())
    }
}

impl From<serde_yaml::Error> for LocaleError {
    fn from(err: serde_yaml::Error) -> Self {
        LocaleError::ParseError(err.to_string())
    }
}

pub struct Locales {
    translations: HashMap<String, HashMap<String, String>>,
    default_locale: String,
}

fn flatten_yaml(value: &serde_yaml::Value, result: &mut HashMap<String, String>, prefix: String) {
    match value {
        serde_yaml::Value::Mapping(map) => {
            for (k, v) in map {
                if let Some(key) = k.as_str() {
                    let new_prefix = if prefix.is_empty() {
                        key.to_string()
                    } else {
                        format!("{}.{}", prefix, key)
                    };
                    flatten_yaml(v, result, new_prefix);
                }
            }
        }
        serde_yaml::Value::String(s) => {
            result.insert(prefix, s.clone());
        }
        _ => {}
    }
}

impl Locales {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, LocaleError> {
        let mut translations = HashMap::new();

        let entries = fs::read_dir(path).map_err(LocaleError::from)?;

        for entry in entries {
            let entry = entry.map_err(LocaleError::from)?;
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "yml") {
                let locale = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| LocaleError::LoadError(path.display().to_string()))?
                    .to_string();

                let content = fs::read_to_string(&path).map_err(LocaleError::from)?;
                let value: serde_yaml::Value = from_str(&content).map_err(|e| {
                    LocaleError::ParseError(format!("Failed to parse {}: {}", path.display(), e))
                })?;

                // Convert nested YAML values to strings
                let mut string_data = HashMap::new();
                if let serde_yaml::Value::Mapping(map) = value {
                    flatten_yaml(&serde_yaml::Value::Mapping(map), &mut string_data, String::new());
                }

                translations.insert(locale, string_data);
            }
        }

        Ok(Self { translations, default_locale: String::new() })
    }

    pub fn set_default(&mut self, locale: &str) -> Result<(), LocaleError> {
        if self.translations.contains_key(locale) {
            self.default_locale = locale.to_string();
            Ok(())
        } else {
            Err(LocaleError::LocaleNotFound(locale.to_string()))
        }
    }

    pub fn t(&self, key: &str) -> String {
        self.translate(&self.default_locale, key).unwrap_or_else(|_| key.to_string())
    }

    pub fn translate(&self, locale: &str, key: &str) -> Result<String, LocaleError> {
        self.translations
            .get(locale)
            .and_then(|data| data.get(key).map(|s| s.to_string()))
            .ok_or_else(|| LocaleError::LocaleNotFound(key.to_string()))
    }
}
