// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Canonical availability shapes (contract e7b70be3, section 3).
use serde::{de::DeserializeOwned, ser::SerializeStruct, Deserialize, Serialize};
use serde_json::Value;

/// Contract e7b70be3, section 3.1. Canonical vocabulary excludes redacted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Availability {
    /// Contract e7b70be3, section 3.1: value carried.
    Present,
    /// Contract e7b70be3, section 3.1: declared absence.
    AbsentByContract,
    /// Contract e7b70be3, section 3.1: source unavailable.
    Unavailable,
    /// Contract e7b70be3, section 3.1: producer raised.
    Errored,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FieldWire {
    availability: Availability,
    value: Value,
    error_code: Value,
}

fn field_bytes<S: serde::Serializer, T: Serialize, E: Serialize>(
    serializer: S,
    availability: Availability,
    value: Option<&T>,
    error: Option<&E>,
) -> Result<S::Ok, S::Error> {
    let mut object = serializer.serialize_struct("Field", 3)?;
    object.serialize_field("availability", &availability)?;
    object.serialize_field("value", &value)?;
    object.serialize_field("error_code", &error)?;
    object.end()
}

// These sums encode the per-field availability subsets, rather than exposing
// independently mutable availability, value, and error members.
macro_rules! field {
    ($name:ident, $( $variant:ident => $availability:ident ),* $(,)?) => {
        /// Contract e7b70be3, section 3.2. A field with a closed availability subset.
        #[derive(Debug, Clone, PartialEq)]
        pub enum $name<T, E> {
            /// Contract e7b70be3, section 3.2: a non-null value and no error.
            Present(T),
            $(
                /// Contract e7b70be3, section 3.2: no value and a closed error.
                $variant(E),
            )*
        }
        impl<T: Serialize, E: Serialize> Serialize for $name<T, E> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                match self {
                    Self::Present(v) => field_bytes::<S, T, E>(serializer, Availability::Present, Some(v), None),
                    $(Self::$variant(e) => field_bytes::<S, T, E>(serializer, Availability::$availability, None, Some(e)),)*
                }
            }
        }
        impl<'de, T: DeserializeOwned, E: DeserializeOwned> Deserialize<'de> for $name<T, E> {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let w = FieldWire::deserialize(d)?;
                match w.availability {
                    Availability::Present if !w.value.is_null() && w.error_code.is_null() =>
                        serde_json::from_value(w.value).map(Self::Present).map_err(serde::de::Error::custom),
                    $(Availability::$availability if w.value.is_null() && !w.error_code.is_null() =>
                        serde_json::from_value(w.error_code).map(Self::$variant).map_err(serde::de::Error::custom),)*
                    _ => Err(serde::de::Error::custom("invalid availability field shape")),
                }
            }
        }
    };
}
field!(PresentUnavailable, Unavailable => Unavailable);
field!(PresentErrored, Errored => Errored);
field!(PresentUnavailableErrored, Unavailable => Unavailable, Errored => Errored);

macro_rules! absent_field {
    ($name:ident, $( $variant:ident => $availability:ident ),* $(,)?) => {
        /// Contract e7b70be3, section 3.2. A field that also permits declared absence.
        #[derive(Debug, Clone, PartialEq)]
        pub enum $name<T, E> {
            /// Contract e7b70be3, section 3.2: a non-null value and no error.
            Present(T),
            /// Contract e7b70be3, section 3.2: neither value nor error.
            AbsentByContract,
            $(
                /// Contract e7b70be3, section 3.2: no value and a closed error.
                $variant(E),
            )*
        }
        impl<T: Serialize, E: Serialize> Serialize for $name<T, E> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                match self {
                    Self::Present(v) => field_bytes::<S, T, E>(serializer, Availability::Present, Some(v), None),
                    Self::AbsentByContract => field_bytes::<S, T, E>(serializer, Availability::AbsentByContract, None, None),
                    $(Self::$variant(e) => field_bytes::<S, T, E>(serializer, Availability::$availability, None, Some(e)),)*
                }
            }
        }
        impl<'de, T: DeserializeOwned, E: DeserializeOwned> Deserialize<'de> for $name<T, E> {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let w = FieldWire::deserialize(d)?;
                match w.availability {
                    Availability::Present if !w.value.is_null() && w.error_code.is_null() =>
                        serde_json::from_value(w.value).map(Self::Present).map_err(serde::de::Error::custom),
                    Availability::AbsentByContract if w.value.is_null() && w.error_code.is_null() => Ok(Self::AbsentByContract),
                    $(Availability::$availability if w.value.is_null() && !w.error_code.is_null() =>
                        serde_json::from_value(w.error_code).map(Self::$variant).map_err(serde::de::Error::custom),)*
                    _ => Err(serde::de::Error::custom("invalid availability field shape")),
                }
            }
        }
    };
}
absent_field!(PresentAbsentErrored, Errored => Errored);
absent_field!(PresentUnavailableAbsent, Unavailable => Unavailable);
absent_field!(PresentUnavailableErroredAbsent, Unavailable => Unavailable, Errored => Errored);

/// Contract e7b70be3, section 3.2. A present or contract-absent field.
#[derive(Debug, Clone, PartialEq)]
pub enum PresentAbsent<T> {
    /// Contract e7b70be3, section 3.2: a non-null value.
    Present(T),
    /// Contract e7b70be3, section 3.2: no value or error.
    AbsentByContract,
}
impl<T: Serialize> Serialize for PresentAbsent<T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Present(v) => field_bytes::<S, T, ()>(s, Availability::Present, Some(v), None),
            Self::AbsentByContract => {
                field_bytes::<S, T, ()>(s, Availability::AbsentByContract, None, None)
            }
        }
    }
}
impl<'de, T: DeserializeOwned> Deserialize<'de> for PresentAbsent<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = FieldWire::deserialize(d)?;
        match w.availability {
            Availability::Present if !w.value.is_null() && w.error_code.is_null() => {
                serde_json::from_value(w.value)
                    .map(Self::Present)
                    .map_err(serde::de::Error::custom)
            }
            Availability::AbsentByContract if w.value.is_null() && w.error_code.is_null() => {
                Ok(Self::AbsentByContract)
            }
            _ => Err(serde::de::Error::custom("invalid availability field shape")),
        }
    }
}

/// Contract e7b70be3, section 3.3. Required slots admit only present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentAvailability {
    /// Contract e7b70be3, section 3.3: required slot.
    Present,
}
/// Contract e7b70be3, section 3.3. Proposal, context, and decision slot shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresentSlot<T> {
    /// Contract e7b70be3, section 3.3.
    pub availability: PresentAvailability,
    /// Contract e7b70be3, section 3.3: always null.
    pub reason: (),
    /// Contract e7b70be3, section 3.3.
    pub digest: String,
    /// Contract e7b70be3, section 3.3.
    pub record: T,
}
/// Contract e7b70be3, section 3.3. Attempt absence vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptAbsentReason {
    /// Contract e7b70be3, section 3.3.
    Blocked,
    /// Contract e7b70be3, section 3.3.
    Unsupported,
    /// Contract e7b70be3, section 3.3.
    NotEnforced,
    /// Contract e7b70be3, section 3.3.
    NoExecutor,
}
/// Contract e7b70be3, section 3.3. Observation and verification absence vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PostAttemptAbsentReason {
    /// Contract e7b70be3, section 3.3.
    AttemptAbsent,
    /// Contract e7b70be3, section 3.3.
    AttemptRejected,
}
/// Contract e7b70be3, section 3.3. Optional lifecycle slot with a typed reason.
#[derive(Debug, Clone, PartialEq)]
pub enum Slot<T, R> {
    /// Contract e7b70be3, section 3.3: digest and record, no reason.
    Present { digest: String, record: T },
    /// Contract e7b70be3, section 3.3: reason, no digest or record.
    AbsentByContract(R),
}
impl<T: Serialize, R: Serialize> Serialize for Slot<T, R> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut object = s.serialize_struct("Slot", 4)?;
        match self {
            Self::Present { digest, record } => {
                object.serialize_field("availability", &Availability::Present)?;
                object.serialize_field("reason", &())?;
                object.serialize_field("digest", digest)?;
                object.serialize_field("record", record)?;
            }
            Self::AbsentByContract(reason) => {
                object.serialize_field("availability", &Availability::AbsentByContract)?;
                object.serialize_field("reason", reason)?;
                object.serialize_field("digest", &())?;
                object.serialize_field("record", &())?;
            }
        }
        object.end()
    }
}
impl<'de, T: DeserializeOwned, R: DeserializeOwned> Deserialize<'de> for Slot<T, R> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            availability: Availability,
            reason: Value,
            digest: Value,
            record: Value,
        }
        let w = Wire::deserialize(d)?;
        match w.availability {
            Availability::Present
                if w.reason.is_null() && !w.digest.is_null() && !w.record.is_null() =>
            {
                Ok(Self::Present {
                    digest: serde_json::from_value(w.digest).map_err(serde::de::Error::custom)?,
                    record: serde_json::from_value(w.record).map_err(serde::de::Error::custom)?,
                })
            }
            Availability::AbsentByContract
                if !w.reason.is_null() && w.digest.is_null() && w.record.is_null() =>
            {
                serde_json::from_value(w.reason)
                    .map(Self::AbsentByContract)
                    .map_err(serde::de::Error::custom)
            }
            _ => Err(serde::de::Error::custom("invalid slot shape")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn required_shape_keys_and_redaction() {
        for bytes in [
            r#"{"availability":"redacted","value":null,"error_code":null}"#,
            r#"{"availability":"present","value":"x"}"#,
            r#"{"availability":"unavailable","value":null,"error_code":null}"#,
            r#"{"availability":"present","value":null,"error_code":null}"#,
        ] {
            assert!(serde_json::from_str::<PresentUnavailable<String, String>>(bytes).is_err());
        }
    }
}
