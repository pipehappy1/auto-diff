#[cfg(feature = "use-serde")]
use serde::{Serialize, Deserialize, Serializer, Deserializer,
	    ser::SerializeStruct,
	    de, de::Visitor, de::SeqAccess, de::MapAccess};
use crate::op::{Op, OpTrait};
use std::fmt;
use std::ops::Deref;



impl Serialize for Box<dyn OpTrait> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer, {
        // 3 is the number of fields in the struct.
        //let mut state = serializer.serialize_struct("OpTrait", 1)?;
        //state.serialize_field("op_name", &self.get_name())?;
        //state.end()
        crate::op::serialize_box::<S>(&self, serializer)
    }
}

impl Serialize for Op {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer, {
        // 3 is the number of fields in the struct.
        let mut state = serializer.serialize_struct("Op", 2)?;
        state.serialize_field("op_name", &self.get_name())?;
	state.serialize_field("op_obj", &self.inner().borrow().deref())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Op {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de>, {

	enum Field { OpName, OpObj }
	
        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where D: Deserializer<'de>, {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("op_name or op_obj")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where E: de::Error, {
                        match value {
                            "op_name" => Ok(Field::OpName),
			    "op_obj" => Ok(Field::OpObj),
                            _ => Err(de::Error::unknown_field(value, &FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }
	
        struct OpVisitor;

        impl<'de> Visitor<'de> for OpVisitor {
            type Value = Op;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Op")
            }

	    fn visit_map<V>(self, mut map: V) -> Result<Op, V::Error>
            where V: MapAccess<'de>, {
		let mut op_name = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::OpName => {
                            if op_name.is_some() {
                                return Err(de::Error::duplicate_field("op_name"));
                            }
                            op_name = Some(map.next_value()?);
                        },
			Field::OpObj => {
                            //if op_obj.is_some() {
                            //    return Err(de::Error::duplicate_field("op_obj"));
                            //}
                            //op_obj = Some(map.next_value()?);
			    let op_name: String = op_name.ok_or_else(|| de::Error::missing_field("op_name"))?;

                            return crate::op::deserialize_map(op_name, map);
                        }
                    }
                }
		Err(de::Error::missing_field("op_obj"))
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Op, V::Error>
            where V: SeqAccess<'de>, {
                let op_name: String = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
		return crate::op::deserialize_seq(op_name, seq);
            }
        }

        const FIELDS: [&str; 2] = ["op_name", "op_obj"];
        deserializer.deserialize_struct("Op", &FIELDS, OpVisitor)
    }
}


#[cfg(all(test, feature = "use-serde"))]
mod tests {
    use crate::op::linear::Linear;
    use super::*;
    use std::rc::Rc;
    use std::cell::RefCell;
    
    #[test]
    fn test_serde_op() {
	let m1 = Linear::new(None, None, true);
	let m1 = Op::new(Rc::new(RefCell::new(Box::new(m1))));
	
        let serialized = serde_pickle::to_vec(&m1, true).unwrap();
        let deserialized: Op = serde_pickle::from_slice(&serialized).unwrap();
        //println!("{:?}", deserialized);
        //assert_eq!(m1, deserialized);
    }
}
