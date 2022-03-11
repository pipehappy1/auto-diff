#[cfg(feature = "use-serde")]
use serde::{Serialize, Deserialize, Serializer, Deserializer,
	    ser::SerializeStruct, ser,
	    de, de::Visitor, de::SeqAccess, de::MapAccess};
use crate::op::{Op, OpTrait};
use std::fmt;
use std::ops::Deref;
use std::rc::Rc;
use std::cell::RefCell;

use crate::op::{View,
                Add, Sub, Mul, Div, Matmul, Outer,
		Linear,
                ELU, ReLU, Sigmoid,
		Conv2d,
                MSELoss, BCEWithLogitsLoss, CrossEntropyLoss,
                Abs, Acos, Asin, Atan, Ceil, Cos, Cosh, Exp, Expm1, Floor, Frac, Log, Log10, Log1p, Log1pexp, Log2, Neg, Reciprocal, Round, Rsqrt, Sign, Sin, Sinh, Sqrt, Tan, Tanh, Trunc,
                MaxPair, MinPair, ArgSort, EqElem, Equal, Ge, Gt, Le, Lt, Ne,
                Cat, Chunk, Gather, IndexSelect, IndexExclude, Reshape, Split, Squeeze, Stack, T, Take, Permute, Unsqueeze, ConditionalSelect, Repeat,
                Det, Inv, NormalizeUnit, Tr,
                Argmax, Argmin, Logsumexp, Mean, Prod, Std, Sum, Variance, Max, Min,
                GetPatch, SetPatch,
};


impl Serialize for Box<dyn OpTrait> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer, {
        // 3 is the number of fields in the struct.
        //let mut state = serializer.serialize_struct("OpTrait", 1)?;
        //state.serialize_field("op_name", &self.get_name())?;
        //state.end()
	match self.get_name() {
	    "View" => {
         let op = self.as_any().downcast_ref::<View>().unwrap();
         return op.serialize(serializer);
         }, 
"Add" => {
         let op = self.as_any().downcast_ref::<Add>().unwrap();
         return op.serialize(serializer);
         }, 
"Sub" => {
         let op = self.as_any().downcast_ref::<Sub>().unwrap();
         return op.serialize(serializer);
         }, 
"Mul" => {
         let op = self.as_any().downcast_ref::<Mul>().unwrap();
         return op.serialize(serializer);
         }, 
"Div" => {
         let op = self.as_any().downcast_ref::<Div>().unwrap();
         return op.serialize(serializer);
         }, 
"Matmul" => {
         let op = self.as_any().downcast_ref::<Matmul>().unwrap();
         return op.serialize(serializer);
         }, 
"Outer" => {
         let op = self.as_any().downcast_ref::<Outer>().unwrap();
         return op.serialize(serializer);
         }, 
"Linear" => {
         let op = self.as_any().downcast_ref::<Linear>().unwrap();
         return op.serialize(serializer);
         }, 
"ELU" => {
         let op = self.as_any().downcast_ref::<ELU>().unwrap();
         return op.serialize(serializer);
         }, 
"ReLU" => {
         let op = self.as_any().downcast_ref::<ReLU>().unwrap();
         return op.serialize(serializer);
         }, 
"Sigmoid" => {
         let op = self.as_any().downcast_ref::<Sigmoid>().unwrap();
         return op.serialize(serializer);
         }, 
"Conv2d" => {
         let op = self.as_any().downcast_ref::<Conv2d>().unwrap();
         return op.serialize(serializer);
         }, 
"MSELoss" => {
         let op = self.as_any().downcast_ref::<MSELoss>().unwrap();
         return op.serialize(serializer);
         }, 
"BCEWithLogitsLoss" => {
         let op = self.as_any().downcast_ref::<BCEWithLogitsLoss>().unwrap();
         return op.serialize(serializer);
         }, 
"CrossEntropyLoss" => {
         let op = self.as_any().downcast_ref::<CrossEntropyLoss>().unwrap();
         return op.serialize(serializer);
         }, 
"Abs" => {
         let op = self.as_any().downcast_ref::<Abs>().unwrap();
         return op.serialize(serializer);
         }, 
"Acos" => {
         let op = self.as_any().downcast_ref::<Acos>().unwrap();
         return op.serialize(serializer);
         }, 
"Asin" => {
         let op = self.as_any().downcast_ref::<Asin>().unwrap();
         return op.serialize(serializer);
         }, 
"Atan" => {
         let op = self.as_any().downcast_ref::<Atan>().unwrap();
         return op.serialize(serializer);
         }, 
"Ceil" => {
         let op = self.as_any().downcast_ref::<Ceil>().unwrap();
         return op.serialize(serializer);
         }, 
"Cos" => {
         let op = self.as_any().downcast_ref::<Cos>().unwrap();
         return op.serialize(serializer);
         }, 
"Cosh" => {
         let op = self.as_any().downcast_ref::<Cosh>().unwrap();
         return op.serialize(serializer);
         }, 
"Exp" => {
         let op = self.as_any().downcast_ref::<Exp>().unwrap();
         return op.serialize(serializer);
         }, 
"Expm1" => {
         let op = self.as_any().downcast_ref::<Expm1>().unwrap();
         return op.serialize(serializer);
         }, 
"Floor" => {
         let op = self.as_any().downcast_ref::<Floor>().unwrap();
         return op.serialize(serializer);
         }, 
"Frac" => {
         let op = self.as_any().downcast_ref::<Frac>().unwrap();
         return op.serialize(serializer);
         }, 
"Log" => {
         let op = self.as_any().downcast_ref::<Log>().unwrap();
         return op.serialize(serializer);
         }, 
"Log10" => {
         let op = self.as_any().downcast_ref::<Log10>().unwrap();
         return op.serialize(serializer);
         }, 
"Log1p" => {
         let op = self.as_any().downcast_ref::<Log1p>().unwrap();
         return op.serialize(serializer);
         }, 
"Log1pexp" => {
         let op = self.as_any().downcast_ref::<Log1pexp>().unwrap();
         return op.serialize(serializer);
         }, 
"Log2" => {
         let op = self.as_any().downcast_ref::<Log2>().unwrap();
         return op.serialize(serializer);
         }, 
"Neg" => {
         let op = self.as_any().downcast_ref::<Neg>().unwrap();
         return op.serialize(serializer);
         }, 
"Reciprocal" => {
         let op = self.as_any().downcast_ref::<Reciprocal>().unwrap();
         return op.serialize(serializer);
         }, 
"Round" => {
         let op = self.as_any().downcast_ref::<Round>().unwrap();
         return op.serialize(serializer);
         }, 
"Rsqrt" => {
         let op = self.as_any().downcast_ref::<Rsqrt>().unwrap();
         return op.serialize(serializer);
         }, 
"Sign" => {
         let op = self.as_any().downcast_ref::<Sign>().unwrap();
         return op.serialize(serializer);
         }, 
"Sin" => {
         let op = self.as_any().downcast_ref::<Sin>().unwrap();
         return op.serialize(serializer);
         }, 
"Sinh" => {
         let op = self.as_any().downcast_ref::<Sinh>().unwrap();
         return op.serialize(serializer);
         }, 
"Sqrt" => {
         let op = self.as_any().downcast_ref::<Sqrt>().unwrap();
         return op.serialize(serializer);
         }, 
"Tan" => {
         let op = self.as_any().downcast_ref::<Tan>().unwrap();
         return op.serialize(serializer);
         }, 
"Tanh" => {
         let op = self.as_any().downcast_ref::<Tanh>().unwrap();
         return op.serialize(serializer);
         }, 
"Trunc" => {
         let op = self.as_any().downcast_ref::<Trunc>().unwrap();
         return op.serialize(serializer);
         }, 
"MaxPair" => {
         let op = self.as_any().downcast_ref::<MaxPair>().unwrap();
         return op.serialize(serializer);
         }, 
"MinPair" => {
         let op = self.as_any().downcast_ref::<MinPair>().unwrap();
         return op.serialize(serializer);
         }, 
"ArgSort" => {
         let op = self.as_any().downcast_ref::<ArgSort>().unwrap();
         return op.serialize(serializer);
         }, 
"EqElem" => {
         let op = self.as_any().downcast_ref::<EqElem>().unwrap();
         return op.serialize(serializer);
         }, 
"Equal" => {
         let op = self.as_any().downcast_ref::<Equal>().unwrap();
         return op.serialize(serializer);
         }, 
"Ge" => {
         let op = self.as_any().downcast_ref::<Ge>().unwrap();
         return op.serialize(serializer);
         }, 
"Gt" => {
         let op = self.as_any().downcast_ref::<Gt>().unwrap();
         return op.serialize(serializer);
         }, 
"Le" => {
         let op = self.as_any().downcast_ref::<Le>().unwrap();
         return op.serialize(serializer);
         }, 
"Lt" => {
         let op = self.as_any().downcast_ref::<Lt>().unwrap();
         return op.serialize(serializer);
         }, 
"Ne" => {
         let op = self.as_any().downcast_ref::<Ne>().unwrap();
         return op.serialize(serializer);
         }, 
"Cat" => {
         let op = self.as_any().downcast_ref::<Cat>().unwrap();
         return op.serialize(serializer);
         }, 
"Chunk" => {
         let op = self.as_any().downcast_ref::<Chunk>().unwrap();
         return op.serialize(serializer);
         }, 
"ConditionalSelect" => {
         let op = self.as_any().downcast_ref::<ConditionalSelect>().unwrap();
         return op.serialize(serializer);
         }, 
"Gather" => {
         let op = self.as_any().downcast_ref::<Gather>().unwrap();
         return op.serialize(serializer);
         }, 
"IndexSelect" => {
         let op = self.as_any().downcast_ref::<IndexSelect>().unwrap();
         return op.serialize(serializer);
         }, 
"IndexExclude" => {
         let op = self.as_any().downcast_ref::<IndexExclude>().unwrap();
         return op.serialize(serializer);
         }, 
"Reshape" => {
         let op = self.as_any().downcast_ref::<Reshape>().unwrap();
         return op.serialize(serializer);
         }, 
"Split" => {
         let op = self.as_any().downcast_ref::<Split>().unwrap();
         return op.serialize(serializer);
         }, 
"Squeeze" => {
         let op = self.as_any().downcast_ref::<Squeeze>().unwrap();
         return op.serialize(serializer);
         }, 
"Stack" => {
         let op = self.as_any().downcast_ref::<Stack>().unwrap();
         return op.serialize(serializer);
         }, 
"T" => {
         let op = self.as_any().downcast_ref::<T>().unwrap();
         return op.serialize(serializer);
         }, 
"Take" => {
         let op = self.as_any().downcast_ref::<Take>().unwrap();
         return op.serialize(serializer);
         }, 
"Permute" => {
         let op = self.as_any().downcast_ref::<Permute>().unwrap();
         return op.serialize(serializer);
         }, 
"Unsqueeze" => {
         let op = self.as_any().downcast_ref::<Unsqueeze>().unwrap();
         return op.serialize(serializer);
         }, 
"Repeat" => {
         let op = self.as_any().downcast_ref::<Repeat>().unwrap();
         return op.serialize(serializer);
         }, 
"Det" => {
         let op = self.as_any().downcast_ref::<Det>().unwrap();
         return op.serialize(serializer);
         }, 
"Inv" => {
         let op = self.as_any().downcast_ref::<Inv>().unwrap();
         return op.serialize(serializer);
         }, 
"NormalizeUnit" => {
         let op = self.as_any().downcast_ref::<NormalizeUnit>().unwrap();
         return op.serialize(serializer);
         }, 
"Tr" => {
         let op = self.as_any().downcast_ref::<Tr>().unwrap();
         return op.serialize(serializer);
         }, 
"Argmax" => {
         let op = self.as_any().downcast_ref::<Argmax>().unwrap();
         return op.serialize(serializer);
         }, 
"Argmin" => {
         let op = self.as_any().downcast_ref::<Argmin>().unwrap();
         return op.serialize(serializer);
         }, 
"Logsumexp" => {
         let op = self.as_any().downcast_ref::<Logsumexp>().unwrap();
         return op.serialize(serializer);
         }, 
"Mean" => {
         let op = self.as_any().downcast_ref::<Mean>().unwrap();
         return op.serialize(serializer);
         }, 
"Prod" => {
         let op = self.as_any().downcast_ref::<Prod>().unwrap();
         return op.serialize(serializer);
         }, 
"Std" => {
         let op = self.as_any().downcast_ref::<Std>().unwrap();
         return op.serialize(serializer);
         }, 
"Sum" => {
         let op = self.as_any().downcast_ref::<Sum>().unwrap();
         return op.serialize(serializer);
         }, 
"Variance" => {
         let op = self.as_any().downcast_ref::<Variance>().unwrap();
         return op.serialize(serializer);
         }, 
"Max" => {
         let op = self.as_any().downcast_ref::<Max>().unwrap();
         return op.serialize(serializer);
         }, 
"Min" => {
         let op = self.as_any().downcast_ref::<Min>().unwrap();
         return op.serialize(serializer);
         }, 
"GetPatch" => {
         let op = self.as_any().downcast_ref::<GetPatch>().unwrap();
         return op.serialize(serializer);
         }, 
"SetPatch" => {
         let op = self.as_any().downcast_ref::<SetPatch>().unwrap();
         return op.serialize(serializer);
         }, 
	    _ => {
		return Err(ser::Error::custom("unknown op"));
	    }
	}
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
			    match op_name.as_str() {
				         "View" => {
             let op_obj: View = Some(map.next_value::<View>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Add" => {
             let op_obj: Add = Some(map.next_value::<Add>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Sub" => {
             let op_obj: Sub = Some(map.next_value::<Sub>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Mul" => {
             let op_obj: Mul = Some(map.next_value::<Mul>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Div" => {
             let op_obj: Div = Some(map.next_value::<Div>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Matmul" => {
             let op_obj: Matmul = Some(map.next_value::<Matmul>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Outer" => {
             let op_obj: Outer = Some(map.next_value::<Outer>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Linear" => {
             let op_obj: Linear = Some(map.next_value::<Linear>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "ELU" => {
             let op_obj: ELU = Some(map.next_value::<ELU>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "ReLU" => {
             let op_obj: ReLU = Some(map.next_value::<ReLU>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Sigmoid" => {
             let op_obj: Sigmoid = Some(map.next_value::<Sigmoid>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Conv2d" => {
             let op_obj: Conv2d = Some(map.next_value::<Conv2d>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "MSELoss" => {
             let op_obj: MSELoss = Some(map.next_value::<MSELoss>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "BCEWithLogitsLoss" => {
             let op_obj: BCEWithLogitsLoss = Some(map.next_value::<BCEWithLogitsLoss>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "CrossEntropyLoss" => {
             let op_obj: CrossEntropyLoss = Some(map.next_value::<CrossEntropyLoss>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Abs" => {
             let op_obj: Abs = Some(map.next_value::<Abs>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Acos" => {
             let op_obj: Acos = Some(map.next_value::<Acos>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Asin" => {
             let op_obj: Asin = Some(map.next_value::<Asin>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Atan" => {
             let op_obj: Atan = Some(map.next_value::<Atan>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Ceil" => {
             let op_obj: Ceil = Some(map.next_value::<Ceil>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Cos" => {
             let op_obj: Cos = Some(map.next_value::<Cos>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Cosh" => {
             let op_obj: Cosh = Some(map.next_value::<Cosh>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Exp" => {
             let op_obj: Exp = Some(map.next_value::<Exp>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Expm1" => {
             let op_obj: Expm1 = Some(map.next_value::<Expm1>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Floor" => {
             let op_obj: Floor = Some(map.next_value::<Floor>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Frac" => {
             let op_obj: Frac = Some(map.next_value::<Frac>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Log" => {
             let op_obj: Log = Some(map.next_value::<Log>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Log10" => {
             let op_obj: Log10 = Some(map.next_value::<Log10>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Log1p" => {
             let op_obj: Log1p = Some(map.next_value::<Log1p>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Log1pexp" => {
             let op_obj: Log1pexp = Some(map.next_value::<Log1pexp>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Log2" => {
             let op_obj: Log2 = Some(map.next_value::<Log2>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Neg" => {
             let op_obj: Neg = Some(map.next_value::<Neg>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Reciprocal" => {
             let op_obj: Reciprocal = Some(map.next_value::<Reciprocal>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Round" => {
             let op_obj: Round = Some(map.next_value::<Round>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Rsqrt" => {
             let op_obj: Rsqrt = Some(map.next_value::<Rsqrt>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Sign" => {
             let op_obj: Sign = Some(map.next_value::<Sign>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Sin" => {
             let op_obj: Sin = Some(map.next_value::<Sin>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Sinh" => {
             let op_obj: Sinh = Some(map.next_value::<Sinh>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Sqrt" => {
             let op_obj: Sqrt = Some(map.next_value::<Sqrt>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Tan" => {
             let op_obj: Tan = Some(map.next_value::<Tan>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Tanh" => {
             let op_obj: Tanh = Some(map.next_value::<Tanh>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Trunc" => {
             let op_obj: Trunc = Some(map.next_value::<Trunc>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "MaxPair" => {
             let op_obj: MaxPair = Some(map.next_value::<MaxPair>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "MinPair" => {
             let op_obj: MinPair = Some(map.next_value::<MinPair>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "ArgSort" => {
             let op_obj: ArgSort = Some(map.next_value::<ArgSort>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "EqElem" => {
             let op_obj: EqElem = Some(map.next_value::<EqElem>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Equal" => {
             let op_obj: Equal = Some(map.next_value::<Equal>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Ge" => {
             let op_obj: Ge = Some(map.next_value::<Ge>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Gt" => {
             let op_obj: Gt = Some(map.next_value::<Gt>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Le" => {
             let op_obj: Le = Some(map.next_value::<Le>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Lt" => {
             let op_obj: Lt = Some(map.next_value::<Lt>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Ne" => {
             let op_obj: Ne = Some(map.next_value::<Ne>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Cat" => {
             let op_obj: Cat = Some(map.next_value::<Cat>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Chunk" => {
             let op_obj: Chunk = Some(map.next_value::<Chunk>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "ConditionalSelect" => {
             let op_obj: ConditionalSelect = Some(map.next_value::<ConditionalSelect>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Gather" => {
             let op_obj: Gather = Some(map.next_value::<Gather>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "IndexSelect" => {
             let op_obj: IndexSelect = Some(map.next_value::<IndexSelect>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "IndexExclude" => {
             let op_obj: IndexExclude = Some(map.next_value::<IndexExclude>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Reshape" => {
             let op_obj: Reshape = Some(map.next_value::<Reshape>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Split" => {
             let op_obj: Split = Some(map.next_value::<Split>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Squeeze" => {
             let op_obj: Squeeze = Some(map.next_value::<Squeeze>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Stack" => {
             let op_obj: Stack = Some(map.next_value::<Stack>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "T" => {
             let op_obj: T = Some(map.next_value::<T>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Take" => {
             let op_obj: Take = Some(map.next_value::<Take>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Permute" => {
             let op_obj: Permute = Some(map.next_value::<Permute>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Unsqueeze" => {
             let op_obj: Unsqueeze = Some(map.next_value::<Unsqueeze>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Repeat" => {
             let op_obj: Repeat = Some(map.next_value::<Repeat>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Det" => {
             let op_obj: Det = Some(map.next_value::<Det>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Inv" => {
             let op_obj: Inv = Some(map.next_value::<Inv>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "NormalizeUnit" => {
             let op_obj: NormalizeUnit = Some(map.next_value::<NormalizeUnit>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Tr" => {
             let op_obj: Tr = Some(map.next_value::<Tr>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Argmax" => {
             let op_obj: Argmax = Some(map.next_value::<Argmax>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Argmin" => {
             let op_obj: Argmin = Some(map.next_value::<Argmin>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Logsumexp" => {
             let op_obj: Logsumexp = Some(map.next_value::<Logsumexp>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Mean" => {
             let op_obj: Mean = Some(map.next_value::<Mean>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Prod" => {
             let op_obj: Prod = Some(map.next_value::<Prod>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Std" => {
             let op_obj: Std = Some(map.next_value::<Std>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Sum" => {
             let op_obj: Sum = Some(map.next_value::<Sum>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Variance" => {
             let op_obj: Variance = Some(map.next_value::<Variance>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Max" => {
             let op_obj: Max = Some(map.next_value::<Max>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "Min" => {
             let op_obj: Min = Some(map.next_value::<Min>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "GetPatch" => {
             let op_obj: GetPatch = Some(map.next_value::<GetPatch>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
         "SetPatch" => {
             let op_obj: SetPatch = Some(map.next_value::<SetPatch>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
            return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
         }, 
				_ => {
				    return Err(de::Error::missing_field("op_obj"));
				}
			    }
                        }
                    }
                }
		Err(de::Error::missing_field("op_obj"))
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Op, V::Error>
            where V: SeqAccess<'de>, {
                let op_name: String = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
		match op_name.as_str() {
		    		    "View" => {
 			let op_obj: View = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Add" => {
 			let op_obj: Add = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Sub" => {
 			let op_obj: Sub = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Mul" => {
 			let op_obj: Mul = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Div" => {
 			let op_obj: Div = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Matmul" => {
 			let op_obj: Matmul = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Outer" => {
 			let op_obj: Outer = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Linear" => {
 			let op_obj: Linear = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "ELU" => {
 			let op_obj: ELU = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "ReLU" => {
 			let op_obj: ReLU = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Sigmoid" => {
 			let op_obj: Sigmoid = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Conv2d" => {
 			let op_obj: Conv2d = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "MSELoss" => {
 			let op_obj: MSELoss = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "BCEWithLogitsLoss" => {
 			let op_obj: BCEWithLogitsLoss = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "CrossEntropyLoss" => {
 			let op_obj: CrossEntropyLoss = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Abs" => {
 			let op_obj: Abs = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Acos" => {
 			let op_obj: Acos = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Asin" => {
 			let op_obj: Asin = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Atan" => {
 			let op_obj: Atan = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Ceil" => {
 			let op_obj: Ceil = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Cos" => {
 			let op_obj: Cos = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Cosh" => {
 			let op_obj: Cosh = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Exp" => {
 			let op_obj: Exp = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Expm1" => {
 			let op_obj: Expm1 = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Floor" => {
 			let op_obj: Floor = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Frac" => {
 			let op_obj: Frac = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Log" => {
 			let op_obj: Log = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Log10" => {
 			let op_obj: Log10 = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Log1p" => {
 			let op_obj: Log1p = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Log1pexp" => {
 			let op_obj: Log1pexp = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Log2" => {
 			let op_obj: Log2 = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Neg" => {
 			let op_obj: Neg = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Reciprocal" => {
 			let op_obj: Reciprocal = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Round" => {
 			let op_obj: Round = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Rsqrt" => {
 			let op_obj: Rsqrt = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Sign" => {
 			let op_obj: Sign = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Sin" => {
 			let op_obj: Sin = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Sinh" => {
 			let op_obj: Sinh = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Sqrt" => {
 			let op_obj: Sqrt = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Tan" => {
 			let op_obj: Tan = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Tanh" => {
 			let op_obj: Tanh = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Trunc" => {
 			let op_obj: Trunc = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "MaxPair" => {
 			let op_obj: MaxPair = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "MinPair" => {
 			let op_obj: MinPair = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "ArgSort" => {
 			let op_obj: ArgSort = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "EqElem" => {
 			let op_obj: EqElem = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Equal" => {
 			let op_obj: Equal = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Ge" => {
 			let op_obj: Ge = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Gt" => {
 			let op_obj: Gt = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Le" => {
 			let op_obj: Le = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Lt" => {
 			let op_obj: Lt = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Ne" => {
 			let op_obj: Ne = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Cat" => {
 			let op_obj: Cat = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Chunk" => {
 			let op_obj: Chunk = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "ConditionalSelect" => {
 			let op_obj: ConditionalSelect = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Gather" => {
 			let op_obj: Gather = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "IndexSelect" => {
 			let op_obj: IndexSelect = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "IndexExclude" => {
 			let op_obj: IndexExclude = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Reshape" => {
 			let op_obj: Reshape = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Split" => {
 			let op_obj: Split = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Squeeze" => {
 			let op_obj: Squeeze = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Stack" => {
 			let op_obj: Stack = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "T" => {
 			let op_obj: T = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Take" => {
 			let op_obj: Take = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Permute" => {
 			let op_obj: Permute = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Unsqueeze" => {
 			let op_obj: Unsqueeze = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Repeat" => {
 			let op_obj: Repeat = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Det" => {
 			let op_obj: Det = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Inv" => {
 			let op_obj: Inv = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "NormalizeUnit" => {
 			let op_obj: NormalizeUnit = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Tr" => {
 			let op_obj: Tr = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Argmax" => {
 			let op_obj: Argmax = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Argmin" => {
 			let op_obj: Argmin = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Logsumexp" => {
 			let op_obj: Logsumexp = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Mean" => {
 			let op_obj: Mean = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Prod" => {
 			let op_obj: Prod = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Std" => {
 			let op_obj: Std = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Sum" => {
 			let op_obj: Sum = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Variance" => {
 			let op_obj: Variance = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Max" => {
 			let op_obj: Max = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "Min" => {
 			let op_obj: Min = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "GetPatch" => {
 			let op_obj: GetPatch = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }

		    "SetPatch" => {
 			let op_obj: SetPatch = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
 			return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
 		    }
		    _ => {
			return Err(de::Error::missing_field("op_obj"));
		    }
		}
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
