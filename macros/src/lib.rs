/// procedure macros

use proc_macro::TokenStream;
use syn::{parse_macro_input, ItemStruct, parse, parse::Parser};
use quote::quote;
use auto_diff::op::OpHandle;

#[proc_macro_attribute]
pub fn op(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut item_struct = parse_macro_input!(input as ItemStruct);
    let _ = parse_macro_input!(args as parse::Nothing);

    if let syn::Fields::Named(ref mut fields) = item_struct.fields {
        fields.named.push(
            syn::Field::parse_named
                .parse2(quote! { handle: OpHandle, })
                .unwrap(),
        );
    }

    return quote! {
        #item_struct
    }
    .into();
}



