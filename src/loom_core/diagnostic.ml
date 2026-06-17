type t = {
  loc : Location.t option;
  message : string;
  details : string list;
}

exception Error of t

let raise_error ?loc ?(details = []) message =
  raise (Error { loc; message; details })

let location_to_string loc =
  let start = loc.Location.loc_start in
  let line = start.Lexing.pos_lnum in
  let col = start.Lexing.pos_cnum - start.Lexing.pos_bol + 1 in
  Printf.sprintf "%s:%d:%d" start.Lexing.pos_fname line col

let to_string { loc; message; details } =
  let head =
    match loc with
    | None -> message
    | Some loc -> Printf.sprintf "%s: %s" (location_to_string loc) message
  in
  match details with
  | [] -> head
  | _ ->
      let rendered =
        details |> List.map (fun line -> "  - " ^ line) |> String.concat "\n"
      in
      head ^ "\n" ^ rendered

let protect f =
  try Ok (f ()) with
  | Error diag -> Error diag

