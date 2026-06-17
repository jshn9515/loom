open Kernel_plan_common

let assign_temp_slots steps =
  let steps_with_ids =
    steps
    |> List.mapi (fun index step ->
           let node_id, output =
             match step with
             | Elementwise step -> (step.node_id, step.output)
             | Reduction step -> (step.node_id, step.output)
           in
           (index, node_id, output))
  in
  let last_use =
    List.fold_left
      (fun acc (index, _, _) ->
        let inputs =
          match List.nth steps index with
          | Elementwise step -> List.map snd step.inputs
          | Reduction step -> (
              match step.source with
              | PlainInput input -> [ input ]
              | MappedInput { inputs; _ } -> List.map snd inputs)
        in
        List.fold_left
          (fun map input_name -> Int_map.add (Hashtbl.hash input_name) index map)
          acc inputs)
      Int_map.empty steps_with_ids
  in
  let next_slot = ref 0 in
  let live = ref [] in
  let slot_map = Hashtbl.create 32 in
  List.iteri
    (fun index step ->
      let still_live, freed =
        List.partition (fun (_, until) -> until >= index) !live
      in
      live := still_live;
      let output =
        match step with
        | Elementwise step -> step.output
        | Reduction step -> step.output
      in
      if not (String.equal output "out") then (
        let key = Hashtbl.hash output in
        let reusable =
          freed
          |> List.sort (fun (_, a) (_, b) -> compare a b)
          |> List.find_opt (fun _ -> true)
          |> Option.map fst
        in
        let slot =
          match reusable with
          | Some slot -> slot
          | None ->
              let slot = !next_slot in
              incr next_slot;
              slot
        in
        let until = Option.value (Int_map.find_opt key last_use) ~default:index in
        Hashtbl.replace slot_map output slot;
        live :=
          (slot, until)
          :: List.filter (fun (existing, _) -> existing <> slot) !live))
    steps;
  (Hashtbl.to_seq slot_map |> List.of_seq, !next_slot)

let rewrite_storage_names steps slot_map =
  let storage_name_of output =
    match List.assoc_opt output slot_map with
    | Some slot -> Printf.sprintf "tmp_slot_%d" slot
    | None -> output
  in
  List.map
    (function
      | Elementwise step ->
          let storage_class =
            match List.assoc_opt step.output slot_map with
            | Some slot -> TemporaryStorage slot
            | None -> OutputStorage
          in
          Elementwise
            {
              step with
              output = storage_name_of step.output;
              inputs =
                List.map
                  (fun (name, source) -> (name, storage_name_of source))
                  step.inputs;
              temp_slot =
                (match storage_class with
                | TemporaryStorage slot -> Some slot
                | OutputStorage -> None);
              storage_class;
            }
      | Reduction step ->
          let storage_class =
            match List.assoc_opt step.output slot_map with
            | Some slot -> TemporaryStorage slot
            | None -> OutputStorage
          in
          let source =
            match step.source with
            | PlainInput input -> PlainInput (storage_name_of input)
            | MappedInput { inputs; scalar_params; body } ->
                MappedInput
                  {
                    inputs =
                      List.map
                        (fun (name, source) -> (name, storage_name_of source))
                        inputs;
                    scalar_params;
                    body;
                  }
          in
          Reduction
            {
              step with
              output = storage_name_of step.output;
              source;
              temp_slot =
                (match storage_class with
                | TemporaryStorage slot -> Some slot
                | OutputStorage -> None);
              storage_class;
            })
    steps
