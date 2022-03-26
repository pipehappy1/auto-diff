/// procedure macros

use proc_macro::TokenStream;
use syn::{parse_macro_input, ItemStruct, parse, parse::Parser};
use quote::quote;
//use auto_diff::op::OpHandle;

#[proc_macro_attribute]
pub fn add_op_handle(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut item_struct = parse_macro_input!(input as ItemStruct);
    let _ = parse_macro_input!(args as parse::Nothing);

    if let syn::Fields::Named(ref mut fields) = item_struct.fields {

        fields.named.push(
            syn::Field::parse_named
                .parse2(quote! {
                    #[cfg_attr(feature = "use-serde", serde(skip))]
                    handle: OpHandle
                })
                .unwrap(),
        );
    }

    return quote! {
        #item_struct
    }
    .into();
}

#[proc_macro_attribute]
pub fn extend_op_impl(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut item_struct = parse_macro_input!(input as ItemStruct);
    let _ = parse_macro_input!(args as parse::Nothing);

    if let syn::Fields::Named(ref mut fields) = item_struct.fields {

        fields.named.push(
            syn::Field::parse_named
                .parse2(quote! {
                    #[cfg_attr(feature = "use-serde", serde(skip))]
                    handle: OpHandle
                })
                .unwrap(),
        );
    }

    return quote! {
        #item_struct
    }
    .into();
}


#[proc_macro]
pub fn gen_serde_funcs(input: TokenStream) -> TokenStream {

    

    let tokens = quote!{
        pub fn serialize<S>(op: &Box<dyn OpTrait>, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer {
            match op.get_name() {
	    "View" => {
                View::serialize::<S>(op, serializer)
            }, 
"Add" => {
         let op = op.as_any().downcast_ref::<Add>().unwrap();
         op.serialize(serializer)
         }, 
"Sub" => {
         let op = op.as_any().downcast_ref::<Sub>().unwrap();
         op.serialize(serializer)
         }, 
"Mul" => {
         let op = op.as_any().downcast_ref::<Mul>().unwrap();
         op.serialize(serializer)
         }, 
"Div" => {
         let op = op.as_any().downcast_ref::<Div>().unwrap();
         op.serialize(serializer)
         }, 
"Matmul" => {
         let op = op.as_any().downcast_ref::<Matmul>().unwrap();
         op.serialize(serializer)
         }, 
"Outer" => {
         let op = op.as_any().downcast_ref::<Outer>().unwrap();
         op.serialize(serializer)
         }, 
"Linear" => {
         let op = op.as_any().downcast_ref::<Linear>().unwrap();
         op.serialize(serializer)
         }, 
"ELU" => {
         let op = op.as_any().downcast_ref::<ELU>().unwrap();
         op.serialize(serializer)
         }, 
"ReLU" => {
         let op = op.as_any().downcast_ref::<ReLU>().unwrap();
         op.serialize(serializer)
         }, 
"Sigmoid" => {
         let op = op.as_any().downcast_ref::<Sigmoid>().unwrap();
         op.serialize(serializer)
         }, 
"Conv2d" => {
         let op = op.as_any().downcast_ref::<Conv2d>().unwrap();
         op.serialize(serializer)
         }, 
"MSELoss" => {
         let op = op.as_any().downcast_ref::<MSELoss>().unwrap();
         op.serialize(serializer)
         }, 
"BCEWithLogitsLoss" => {
         let op = op.as_any().downcast_ref::<BCEWithLogitsLoss>().unwrap();
         op.serialize(serializer)
         }, 
"CrossEntropyLoss" => {
         let op = op.as_any().downcast_ref::<CrossEntropyLoss>().unwrap();
         op.serialize(serializer)
         }, 
"Abs" => {
         let op = op.as_any().downcast_ref::<Abs>().unwrap();
         op.serialize(serializer)
         }, 
"Acos" => {
         let op = op.as_any().downcast_ref::<Acos>().unwrap();
         op.serialize(serializer)
         }, 
"Asin" => {
         let op = op.as_any().downcast_ref::<Asin>().unwrap();
         op.serialize(serializer)
         }, 
"Atan" => {
         let op = op.as_any().downcast_ref::<Atan>().unwrap();
         op.serialize(serializer)
         }, 
"Ceil" => {
         let op = op.as_any().downcast_ref::<Ceil>().unwrap();
         op.serialize(serializer)
         }, 
"Cos" => {
         let op = op.as_any().downcast_ref::<Cos>().unwrap();
         op.serialize(serializer)
         }, 
"Cosh" => {
         let op = op.as_any().downcast_ref::<Cosh>().unwrap();
         op.serialize(serializer)
         }, 
"Exp" => {
         let op = op.as_any().downcast_ref::<Exp>().unwrap();
         op.serialize(serializer)
         }, 
"Expm1" => {
         let op = op.as_any().downcast_ref::<Expm1>().unwrap();
         op.serialize(serializer)
         }, 
"Floor" => {
         let op = op.as_any().downcast_ref::<Floor>().unwrap();
         op.serialize(serializer)
         }, 
"Frac" => {
         let op = op.as_any().downcast_ref::<Frac>().unwrap();
         op.serialize(serializer)
         }, 
"Log" => {
         let op = op.as_any().downcast_ref::<Log>().unwrap();
         op.serialize(serializer)
         }, 
"Log10" => {
         let op = op.as_any().downcast_ref::<Log10>().unwrap();
         op.serialize(serializer)
         }, 
"Log1p" => {
         let op = op.as_any().downcast_ref::<Log1p>().unwrap();
         op.serialize(serializer)
         }, 
"Log1pexp" => {
         let op = op.as_any().downcast_ref::<Log1pexp>().unwrap();
         op.serialize(serializer)
         }, 
"Log2" => {
         let op = op.as_any().downcast_ref::<Log2>().unwrap();
         op.serialize(serializer)
         }, 
"Neg" => {
         let op = op.as_any().downcast_ref::<Neg>().unwrap();
         op.serialize(serializer)
         }, 
"Reciprocal" => {
         let op = op.as_any().downcast_ref::<Reciprocal>().unwrap();
         op.serialize(serializer)
         }, 
"Round" => {
         let op = op.as_any().downcast_ref::<Round>().unwrap();
         op.serialize(serializer)
         }, 
"Rsqrt" => {
         let op = op.as_any().downcast_ref::<Rsqrt>().unwrap();
         op.serialize(serializer)
         }, 
"Sign" => {
         let op = op.as_any().downcast_ref::<Sign>().unwrap();
         op.serialize(serializer)
         }, 
"Sin" => {
         let op = op.as_any().downcast_ref::<Sin>().unwrap();
         op.serialize(serializer)
         }, 
"Sinh" => {
         let op = op.as_any().downcast_ref::<Sinh>().unwrap();
         op.serialize(serializer)
         }, 
"Sqrt" => {
         let op = op.as_any().downcast_ref::<Sqrt>().unwrap();
         op.serialize(serializer)
         }, 
"Tan" => {
         let op = op.as_any().downcast_ref::<Tan>().unwrap();
         op.serialize(serializer)
         }, 
"Tanh" => {
         let op = op.as_any().downcast_ref::<Tanh>().unwrap();
         op.serialize(serializer)
         }, 
"Trunc" => {
         let op = op.as_any().downcast_ref::<Trunc>().unwrap();
         op.serialize(serializer)
         }, 
"MaxPair" => {
         let op = op.as_any().downcast_ref::<MaxPair>().unwrap();
         op.serialize(serializer)
         }, 
"MinPair" => {
         let op = op.as_any().downcast_ref::<MinPair>().unwrap();
         op.serialize(serializer)
         }, 
"ArgSort" => {
         let op = op.as_any().downcast_ref::<ArgSort>().unwrap();
         op.serialize(serializer)
         }, 
"EqElem" => {
         let op = op.as_any().downcast_ref::<EqElem>().unwrap();
         op.serialize(serializer)
         }, 
"Equal" => {
         let op = op.as_any().downcast_ref::<Equal>().unwrap();
         op.serialize(serializer)
         }, 
"Ge" => {
         let op = op.as_any().downcast_ref::<Ge>().unwrap();
         op.serialize(serializer)
         }, 
"Gt" => {
         let op = op.as_any().downcast_ref::<Gt>().unwrap();
         op.serialize(serializer)
         }, 
"Le" => {
         let op = op.as_any().downcast_ref::<Le>().unwrap();
         op.serialize(serializer)
         }, 
"Lt" => {
         let op = op.as_any().downcast_ref::<Lt>().unwrap();
         op.serialize(serializer)
         }, 
"Ne" => {
         let op = op.as_any().downcast_ref::<Ne>().unwrap();
         op.serialize(serializer)
         }, 
"Cat" => {
         let op = op.as_any().downcast_ref::<Cat>().unwrap();
         op.serialize(serializer)
         }, 
"Chunk" => {
         let op = op.as_any().downcast_ref::<Chunk>().unwrap();
         op.serialize(serializer)
         }, 
"ConditionalSelect" => {
         let op = op.as_any().downcast_ref::<ConditionalSelect>().unwrap();
         op.serialize(serializer)
         }, 
"Gather" => {
         let op = op.as_any().downcast_ref::<Gather>().unwrap();
         op.serialize(serializer)
         }, 
"IndexSelect" => {
         let op = op.as_any().downcast_ref::<IndexSelect>().unwrap();
         op.serialize(serializer)
         }, 
"IndexExclude" => {
         let op = op.as_any().downcast_ref::<IndexExclude>().unwrap();
         op.serialize(serializer)
         }, 
"Reshape" => {
         let op = op.as_any().downcast_ref::<Reshape>().unwrap();
         op.serialize(serializer)
         }, 
"Split" => {
         let op = op.as_any().downcast_ref::<Split>().unwrap();
         op.serialize(serializer)
         }, 
"Squeeze" => {
         let op = op.as_any().downcast_ref::<Squeeze>().unwrap();
         op.serialize(serializer)
         }, 
"Stack" => {
         let op = op.as_any().downcast_ref::<Stack>().unwrap();
         op.serialize(serializer)
         }, 
"T" => {
         let op = op.as_any().downcast_ref::<T>().unwrap();
         op.serialize(serializer)
         }, 
"Take" => {
         let op = op.as_any().downcast_ref::<Take>().unwrap();
         op.serialize(serializer)
         }, 
"Permute" => {
         let op = op.as_any().downcast_ref::<Permute>().unwrap();
         op.serialize(serializer)
         }, 
"Unsqueeze" => {
         let op = op.as_any().downcast_ref::<Unsqueeze>().unwrap();
         op.serialize(serializer)
         }, 
"Repeat" => {
         let op = op.as_any().downcast_ref::<Repeat>().unwrap();
         op.serialize(serializer)
         }, 
"Det" => {
         let op = op.as_any().downcast_ref::<Det>().unwrap();
         op.serialize(serializer)
         }, 
"Inv" => {
         let op = op.as_any().downcast_ref::<Inv>().unwrap();
         op.serialize(serializer)
         }, 
"NormalizeUnit" => {
         let op = op.as_any().downcast_ref::<NormalizeUnit>().unwrap();
         op.serialize(serializer)
         }, 
"Tr" => {
         let op = op.as_any().downcast_ref::<Tr>().unwrap();
         op.serialize(serializer)
         }, 
"Argmax" => {
         let op = op.as_any().downcast_ref::<Argmax>().unwrap();
         op.serialize(serializer)
         }, 
"Argmin" => {
         let op = op.as_any().downcast_ref::<Argmin>().unwrap();
         op.serialize(serializer)
         }, 
"Logsumexp" => {
         let op = op.as_any().downcast_ref::<Logsumexp>().unwrap();
         op.serialize(serializer)
         }, 
"Mean" => {
         let op = op.as_any().downcast_ref::<Mean>().unwrap();
         op.serialize(serializer)
         }, 
"Prod" => {
         let op = op.as_any().downcast_ref::<Prod>().unwrap();
         op.serialize(serializer)
         }, 
"Std" => {
         let op = op.as_any().downcast_ref::<Std>().unwrap();
         op.serialize(serializer)
         }, 
"Sum" => {
         let op = op.as_any().downcast_ref::<Sum>().unwrap();
         op.serialize(serializer)
         }, 
"Variance" => {
         let op = op.as_any().downcast_ref::<Variance>().unwrap();
         op.serialize(serializer)
         }, 
"Max" => {
         let op = op.as_any().downcast_ref::<Max>().unwrap();
         op.serialize(serializer)
         }, 
"Min" => {
         let op = op.as_any().downcast_ref::<Min>().unwrap();
         op.serialize(serializer)
         }, 
"GetPatch" => {
         let op = op.as_any().downcast_ref::<GetPatch>().unwrap();
         op.serialize(serializer)
         }, 
"SetPatch" => {
         let op = op.as_any().downcast_ref::<SetPatch>().unwrap();
         op.serialize(serializer)
         }, 
	    other => {
		return Err(ser::Error::custom(format!("unknown op {:?}", other)));
	    }
	}
        }
    };

    
    tokens.into()
}


#[cfg(test)]
mod tests {
    
    #[test]
    fn test() {
        
    }
}
