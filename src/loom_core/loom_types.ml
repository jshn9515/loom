type stage_type =
  | LLFloat
  | LLBool
  | LLTensor1F32
  | LLUnit
  | LLTuple of stage_type list
  | LLFun of stage_type list * stage_type

type entry_param_kind =
  | ScalarF32
  | Tensor1F32

let rec stage_type_to_string = function
  | LLFloat -> "float"
  | LLBool -> "bool"
  | LLTensor1F32 -> "tensor1<f32>"
  | LLUnit -> "unit"
  | LLTuple items ->
      Printf.sprintf "(%s)"
        (String.concat " * " (List.map stage_type_to_string items))
  | LLFun (params, result) ->
      Printf.sprintf "(%s -> %s)"
        (String.concat " -> " (List.map stage_type_to_string params))
        (stage_type_to_string result)

let option_all values =
  let rec loop acc = function
    | [] -> Some (List.rev acc)
    | Some value :: rest -> loop (value :: acc) rest
    | None :: _ -> None
  in
  loop [] values

let stage_type_of_string text =
  let text = String.trim text in
  let rec split_top_level ~sep value =
    let rec loop depth start index acc =
      if index = String.length value then
        List.rev (String.sub value start (index - start) :: acc)
      else
        match value.[index] with
        | '(' -> loop (depth + 1) start (index + 1) acc
        | ')' -> loop (depth - 1) start (index + 1) acc
        | ch when ch = sep && depth = 0 ->
            let item = String.sub value start (index - start) in
            loop depth (index + 1) (index + 1) (item :: acc)
        | _ -> loop depth start (index + 1) acc
    in
    loop 0 0 0 []
    |> List.map String.trim
    |> List.filter (fun item -> item <> "")
  in
  let rec parse value =
    let value = String.trim value in
    match value with
    | "float" -> Some LLFloat
    | "bool" -> Some LLBool
    | "tensor1<f32>" -> Some LLTensor1F32
    | "unit" -> Some LLUnit
    | _ when String.length value >= 2 && value.[0] = '(' && value.[String.length value - 1] = ')' ->
        let inner = String.sub value 1 (String.length value - 2) |> String.trim in
        let arrow_parts = split_top_level ~sep:'-' inner in
        if List.length arrow_parts > 1 && String.contains inner '>' then
          let arrow_parts =
            let parts = ref [] in
            let buffer = Buffer.create (String.length inner) in
            let depth = ref 0 in
            let flush () =
              let item = Buffer.contents buffer |> String.trim in
              Buffer.clear buffer;
              if item <> "" then parts := item :: !parts
            in
            let i = ref 0 in
            while !i < String.length inner do
              let ch = inner.[!i] in
              begin
                match ch with
                | '(' ->
                    incr depth;
                    Buffer.add_char buffer ch
                | ')' ->
                    decr depth;
                    Buffer.add_char buffer ch
                | '-' when !depth = 0 && !i + 1 < String.length inner && inner.[!i + 1] = '>' ->
                    flush ();
                    incr i
                | _ -> Buffer.add_char buffer ch
              end;
              incr i
            done;
            flush ();
            List.rev !parts
          in
          match List.rev arrow_parts with
          | [] -> None
          | result :: rev_params -> (
              match option_all (List.map parse (List.rev rev_params)), parse result with
              | Some params, Some result -> Some (LLFun (params, result))
              | _ -> None )
        else
          let tuple_parts = split_top_level ~sep:'*' inner in
          if List.length tuple_parts <= 1 then None
          else option_all (List.map parse tuple_parts) |> Option.map (fun items -> LLTuple items)
    | _ -> None
  in
  parse text

let entry_param_kind_to_string = function
  | ScalarF32 -> "scalar-f32"
  | Tensor1F32 -> "tensor1-f32"

let rec classify_type ~tensor_type ty =
  match (Types.Transient_expr.repr ty).Types.desc with
  | Types.Tconstr (path, [], _) when Path.same path Predef.path_float -> Some LLFloat
  | Types.Tconstr (path, [], _) when Path.same path Predef.path_bool -> Some LLBool
  | Types.Tconstr (path, [], _) when Path.same path Predef.path_unit -> Some LLUnit
  | Types.Tconstr (path, [], _)
    when Path.same path tensor_type
         || String.equal (Path.name path) (Path.name tensor_type)
         || String.ends_with ~suffix:"Tensor1.t" (Path.name path) ->
      Some LLTensor1F32
  | Types.Ttuple labels ->
      labels
      |> List.map snd
      |> List.map (classify_type ~tensor_type)
      |> option_all
      |> Option.map (fun items -> LLTuple items)
  | Types.Tarrow (_, lhs, rhs, _) -> (
      let rec collect acc ty =
        match (Types.Transient_expr.repr ty).Types.desc with
        | Types.Tarrow (_, lhs, rhs, _) -> collect (lhs :: acc) rhs
        | _ -> (List.rev acc, ty)
      in
      let params, result = collect [ lhs ] rhs in
      match
        option_all (List.map (classify_type ~tensor_type) params),
        classify_type ~tensor_type result
      with
      | Some params, Some result -> Some (LLFun (params, result))
      | _ -> None )
  | _ -> None
