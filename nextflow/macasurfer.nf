nextflow.enable.dsl=2

// —— 自动探测 GPU 索引序列 ——
// 优先用调度/环境提供的信息，其次再尝试 nvidia-smi
def inferGpuIndices() {
    def env = System.getenv()

    // 1) 优先用 CUDA_VISIBLE_DEVICES（容器/SLURM 通常会设置）
    def cvd = env.get('CUDA_VISIBLE_DEVICES')
    if (cvd && cvd.trim()) {
        def lst = cvd.split(',').collect { it.trim() }.findAll { it }
        def indices = lst.collect { it.isInteger() ? it.toInteger() : it }
        log.info "CUDA_VISIBLE_DEVICES=${cvd}  →  GPU 索引: ${indices}"
        return indices
    }

    // 2) 其次看 SLURM（某些站点会暴露 JOB/STEP 的 GPU 列表）
    def sg = env.get('SLURM_JOB_GPUS') ?: env.get('SLURM_STEP_GPUS')
    if (sg && sg.trim()) {
        def lst = sg.split(',').collect { it.trim() }.findAll { it }
        def indices = lst.collect { it.isInteger() ? it.toInteger() : it }
        log.info "SLURM_*_GPUS=${sg}  →  GPU 索引: ${indices}"
        return indices
    }

    // 3) 兜底：用 nvidia-smi 检测 GPU 数量并生成 [0,1,2,...]
    try {
        def p = ['bash', '-lc', 'nvidia-smi -L | wc -l'].execute()
        p.waitForOrKill(2000)
        def n = p.text.trim()
        if (n.isInteger() && n.toInteger() > 0) {
            def count = n.toInteger()
            def indices = (0..<count).toList()
            log.info "nvidia-smi 检测到 ${count} GPU(s) → 索引: ${indices}"
            return indices
        }
    } catch (ignored) { /* no-op */ }

    // 4) 无法检测到则默认返回 [0]
    log.warn "未检测到显式 GPU 信息，默认返回 [0]"
    return [0]
}



// ---------------------------------------
// 1) YAML -> runs.tsv + sessions.tsv
// ---------------------------------------
process yaml_to_tsv {
    cpus 1

    input:
      val qc_root
      val participant_label
      val session_id

    output:
      path "runs.tsv"
      path "sessions.tsv"

    script:
    """
    ${params.python_inter} - << 'PY'
    import yaml
    from pathlib import Path

    qc_dir = Path("${qc_root}").expanduser().resolve()

    def _norm(v):
      if v is None:
        return None
      s = str(v).strip()
      if s == "" or s.lower() == "null":
        return None
      return s

    def _to_list(v):
      v = _norm(v)
      if v is None:
        return None
      if v.startswith('[') and v.endswith(']'):
        v = v[1:-1].strip()
      parts = [x.strip() for x in v.split(',') if x.strip()]
      return parts if parts else None

    participants = _to_list("${participant_label}")
    sessions     = _to_list("${session_id}")

    if not qc_dir.exists():
      raise FileNotFoundError(f"qc_dir not found: {qc_dir}")

    yaml_paths = []

    if participants:
      for sub in participants:
        sub_dir = qc_dir / sub
        if not sub_dir.exists():
          continue
        if sessions:
          for ses in sessions:
            p = sub_dir / ses / "orig_config.yaml"
            if p.exists():
              yaml_paths.append(p)
        else:
          yaml_paths.extend(sorted(sub_dir.glob("ses-*/orig_config.yaml")))
    else:
      if sessions:
        for ses in sessions:
          yaml_paths.extend(sorted(qc_dir.glob(f"sub-*/{ses}/orig_config.yaml")))
      else:
        yaml_paths = sorted(qc_dir.glob("sub-*/ses-*/orig_config.yaml"))

    yaml_paths = [p for p in yaml_paths if p.is_file()]

    if not yaml_paths:
      raise RuntimeError(
        "No per-session YAMLs found under qc_dir. "
        f"qc_dir={qc_dir}, participants={participants or 'ANY'}, sessions={sessions or 'ANY'}"
      )

    merged = {}
    for yp in yaml_paths:
      y = yaml.safe_load(yp.read_text()) or {}
      for subj, ses_dict in y.items():
        merged.setdefault(subj, {})
        for ses, mod_dict in (ses_dict or {}).items():
          if sessions and ses not in sessions:
            continue
          merged[subj].setdefault(ses, {})
          for mod in ("T1","T2","FLAIR"):
            merged[subj][ses].setdefault(mod, {})
            runs = (mod_dict or {}).get(mod, {}) or {}
            merged[subj][ses][mod].update(runs)

    run_lines = []
    sess_lines = []

    def b(v): return "true" if bool(v) else "false"

    for subj in sorted(merged.keys()):
      for ses in sorted((merged[subj] or {}).keys()):
        mod_dict = merged[subj][ses] or {}
        ref = {"T1":"", "T2":"", "FLAIR":""}

        for modality in ["T1","T2","FLAIR"]:
          runs = (mod_dict or {}).get(modality, {}) or {}
          for run_name in sorted(runs.keys()):
            info = runs[run_name] or {}
            if info.get("is_refer"):
              ref[modality] = run_name
            run_lines.append("\\t".join([
              subj, ses, modality, run_name,
              str(info.get("orig","")),
              b(info.get("is_brain", False)),
              b(info.get("good_mask", False)),
              b(info.get("is_refer", False)),
              b(info.get("good_quality", True)),
            ]))

        sess_lines.append("\\t".join([subj, ses, ref["T1"], ref["T2"], ref["FLAIR"]]))

    Path("runs.tsv").write_text("\\n".join(run_lines) + ("\\n" if run_lines else ""))
    Path("sessions.tsv").write_text("\\n".join(sess_lines) + ("\\n" if sess_lines else ""))

    print(f"[yaml_to_tsv] qc_dir={qc_dir}  matched_yaml_count={len(yaml_paths)}")
    PY
    """
}


// ---------------------------------------
// 2) Init per-session dirs (fixed layout)
// ---------------------------------------
process init_session_dirs {
    tag { "${subj}/${ses}" }
    cpus 1
    memory '200 MB'

    input:
      tuple val(subj), val(ses), val(out_root)

    output:
      tuple val(ses_key), val(subj), val(ses),
            val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir)

    script:
    prepare_dir = "${out_root}/${subj}/${ses}/Prepare"
    enhance_dir = "${out_root}/${subj}/${ses}/Enhance"
    surface_dir = "${out_root}/${subj}/${ses}/Surface"
    resample_dir= "${out_root}/${subj}/${ses}/Resample"
    ses_key = "${subj}::${ses}"
    """
    set -euo pipefail
    mkdir -p "${prepare_dir}" "${enhance_dir}" "${surface_dir}" "${resample_dir}"
    """
}

// ---------------------------------------
// Workflow: GET_INFO
// ---------------------------------------
workflow info {
  take:
    qc_dir
    out_root

  main:
    (runs_tsv, sess_tsv) = yaml_to_tsv(qc_dir, params.participant_label, params.session_id)

    // --------------------
    // 1) session-level ctx (one row per subj+ses)
    // --------------------
    sess_keyed = sess_tsv
      .splitText()
      .map { line ->
        def c = line.trim().split('\t')
        def subj = c[0]
        def ses  = c[1]
        def refT1 = c.size() > 2 ? c[2] : ""
        def refT2 = c.size() > 3 ? c[3] : ""
        def refFL = c.size() > 4 ? c[4] : ""
        def session_key = "${subj}::${ses}"

        // 规范化一下（防 CRLF）
        refT1 = refT1?.toString()?.trim()?.replace('\r','') ?: ""
        refT2 = refT2?.toString()?.trim()?.replace('\r','') ?: ""
        refFL = refFL?.toString()?.trim()?.replace('\r','') ?: ""

        tuple(session_key, [subj:subj, ses:ses, refT1:refT1, refT2:refT2, refFL:refFL])
      }

    // --------------------
    // 2) init dirs per session
    // --------------------
    dirs_raw = init_session_dirs(
      sess_keyed.map { session_key, sm -> tuple(sm.subj, sm.ses, out_root) }
    )

    // NOTE: 假设 dirs_raw = (session_key, subj, ses, prepare, enhance, surface, resample)
    dirs_keyed = dirs_raw.map { row ->
      def session_key = row[0]
      tuple(session_key, [
        prepare_dir: row[3],
        enhance_dir: row[4],
        surface_dir: row[5],
        resample_dir: row[6],
      ])
    }

    // ctx_keyed: (session_key, ctxMap)
    ctx_keyed = sess_keyed.join(dirs_keyed)
      .map { session_key, sm, dm ->
        tuple(session_key, sm + dm)
      }

    // --------------------
    // 3) run-level records keyed by session_key (for attaching ctx)
    // --------------------
    runs_sess_keyed = runs_tsv
      .splitText()
      .map { line ->
        def c = line.trim().split('\t')
        def subj = c[0]
        def ses  = c[1]
        def modality = c[2]
        def run  = c[3]
        def orig = c[4]
        def is_brain = c[5]
        def good_mask = c[6]
        def is_refer  = c[7]
        def good_quality = c[8]

        // normalize
        modality = modality?.toString()?.trim()?.replace('\r','') ?: ""
        run      = run?.toString()?.trim()?.replace('\r','') ?: ""
        orig     = orig?.toString()?.trim()?.replace('\r','') ?: ""
        is_brain = is_brain?.toString()?.trim()?.replace('\r','') ?: "false"
        good_mask= good_mask?.toString()?.trim()?.replace('\r','') ?: "false"
        is_refer = is_refer?.toString()?.trim()?.replace('\r','') ?: "false"
        good_quality = good_quality?.toString()?.trim()?.replace('\r','') ?: "true"

        def session_key = "${subj}::${ses}"
        tuple(session_key, [
          subj:subj, ses:ses, modality:modality, run:run, orig:orig,
          is_brain:is_brain, good_mask:good_mask, is_refer:is_refer, good_quality:good_quality
        ])
      }

    // --------------------
    // 4) attach ctx to each run (IMPORTANT: use combine to broadcast ctx)
    // --------------------
    run_ctx_by_sess = runs_sess_keyed
      .combine(ctx_keyed, by: 0)
      .map { session_key, rm, cx ->
        tuple(session_key, rm + cx)
      }
    

    // promote to run_key: (run_key, runCtxMap)
    run_ctx_keyed = run_ctx_by_sess.map { session_key, m ->
      def run_key = "${m.subj}::${m.ses}::${m.modality}::${m.run}"
      tuple(run_key, m)
    }

  emit:
    ctx_keyed
    run_ctx_keyed
    run_ctx_by_sess
}

// ---------------------------------------
// 3) Skullstrip (real / pseudo)
// ---------------------------------------
process anat_skullstrip {
    tag { "${subj}/${ses}/${modality}/${run}" }
    accelerator 1
    maxForks 1 * ((params.per_gpu == null) ? 1 : (params.per_gpu as int))

    input:
      tuple val(run_key), val(subj), val(ses),
            val(modality), val(run), val(orig),
            val(is_brain), val(good_mask), val(is_refer), val(good_quality),
            val(prepare_dir)

    output:
      tuple val(run_key), val(done)

    script:
    done = 1
    predict_script = "${params.maca_brainnet_dir}/predict_ensemble.py"
    skull_model_dir = "${params.maca_brainnet_dir}/swinunetr_models/skull_stripping"
    if ( params.process_stage == 'all' || params.process_stage == 'prepare' )
      """
      set -euo pipefail
      mkdir -p "${prepare_dir}/Volume/${modality}/${run}"
      ${params.python_inter} ${params.prepare_script_dir}/scripts/brainnet_skullstrip.py \
        --input ${orig} \
        --output-dir ${prepare_dir}/Volume/${modality}/${run} \
        --model-dir ${skull_model_dir} \
        --predict-script ${predict_script} \
        --python-inter ${params.python_inter} \
        --is-brain ${is_brain}
      """
      else
        """echo skip prepare"""
}

// -------------------------
// 3c) Check brainmask
// -------------------------
process qc_skullstrip {
    tag { "${subj}/${ses}/${modality}/${run}" }
    cpus 1

    input:
      tuple val(run_key), val(subj), val(ses),
            val(modality), val(run), val(prepare_dir)
      val qc_dir

    output:
      tuple val(run_key), val(qc_done)

    script:
    qc_done = 1
    head_path = "${prepare_dir}/Volume/${modality}/${run}/head.nii.gz"
    brainmask_path = "${prepare_dir}/Volume/${modality}/${run}/brainmask.nii.gz"
    session_qc_dir = "${qc_dir}/${subj}/${ses}"
    qc_output = "${session_qc_dir}/qc_skullstrip_${modality}_${run}.png"
    if ( params.process_stage == 'all' || params.process_stage == 'prepare' )
      """
      set -euo pipefail
      mkdir -p "${session_qc_dir}"

      if [[ -f "${head_path}" && -f "${brainmask_path}" ]]; then
        ${params.python_inter} ${params.utils_path}/qc_skullstrip.py \
          --head "${head_path}" \
          --brainmask "${brainmask_path}" \
          --output "${qc_output}" \
          --rows ${params.qc_grid_rows} --cols ${params.qc_grid_cols}
        echo "[OK] QC skullstrip saved to ${qc_output}"
      else
        echo "[WARN] Missing head or brainmask, skip QC for ${run}"
      fi
      """
    else
        """echo skip prepare"""
}

// ---------------------------------------
// 4) Fix brainmask (needs session ref)
//    - only when good_mask=false
// ---------------------------------------
process anat_fix_brainmask {
    tag { "${subj}/${ses}/${modality}/${run}" }
    cpus 1

    input:
      tuple val(run_key), val(subj), val(ses), val(modality), val(run),
            val(good_mask), val(refT1), val(refT2), val(refFLAIR),
            val(prepare_dir)

    output:
      tuple val(run_key), val(done)

    script:
    done = 1
    if ( params.process_stage == 'all' || params.process_stage == 'prepare' )
      """
      set -euo pipefail

      # -------- Nextflow-substituted constants --------
      GM="${good_mask}"
      MOD="${modality}"
      RUN="${run}"
      PREP="${prepare_dir}"
      RT1="${refT1}"
      RT2="${refT2}"
      RFL="${refFLAIR}"

      # normalize (avoid CRLF/space issues)
      GM="\$(echo -n "\$GM"  | tr -d '\\r' | xargs)"
      MOD="\$(echo -n "\$MOD" | tr -d '\\r' | xargs)"

      echo "[DBG] fix_brainmask subj=${subj} ses=${ses} MOD=\$MOD RUN=\$RUN GM=\$GM"
      echo "[DBG] refs: T1=\$RT1 T2=\$RT2 FLAIR=\$RFL"

      if [[ "\$GM" == "true" ]]; then
        echo "[INFO] good_mask=true, skip fix"
        exit 0
      fi

      # -------- bash runtime vars (must NOT be groovy-interpolated) --------
      ref=""
      refmod="\$MOD"

      # same-modality ref first
      if [[ "\$MOD" == "T1" ]]; then
        ref="\$RT1"
      elif [[ "\$MOD" == "T2" ]]; then
        ref="\$RT2"
      elif [[ "\$MOD" == "FLAIR" ]]; then
        ref="\$RFL"
      else
        echo "[WARN] unknown modality='\$MOD'"
      fi

      # fallback any modality
      if [[ -z "\$ref" ]]; then
        if [[ -n "\$RT1" ]]; then ref="\$RT1"; refmod="T1"; fi
        if [[ -z "\$ref" && -n "\$RT2" ]]; then ref="\$RT2"; refmod="T2"; fi
        if [[ -z "\$ref" && -n "\$RFL" ]]; then ref="\$RFL"; refmod="FLAIR"; fi
      fi

      echo "[DBG] picked refmod=\$refmod ref=\$ref"

      if [[ -z "\$ref" ]]; then
        echo "[WARN] no reference found, skip fix"
        exit 0
      fi

      if [[ ! -d "\$PREP/Volume/\$refmod/\$ref" ]]; then
        echo "[WARN] reference dir missing: \$PREP/Volume/\$refmod/\$ref"
        exit 0
      fi

      ${params.python_inter} ${params.prepare_script_dir}/scripts/fix_brainmask.py \
        --run_name "\$RUN" \
        --refer_name "\$ref" \
        --run_dir "\$PREP/Volume/\$MOD/\$RUN" \
        --refer_dir "\$PREP/Volume/\$refmod/\$ref" \
        --python_inter "${params.python_inter}" \
        --utils_path "${params.utils_path}"
      """
    else
        """echo skip prepare"""
}

// ---------------------------------------
// 4b) Check fixed brainmask
// ---------------------------------------
process qc_fixed_brainmask {
    tag { "${subj}/${ses}/${modality}/${run}" }
    cpus 1

    input:
      tuple val(run_key), val(subj), val(ses),
            val(modality), val(run), val(good_mask), val(prepare_dir)
      val qc_dir

    output:
      tuple val(run_key), val(qc_done)

    script:
    qc_done = 1
    head_path = "${prepare_dir}/Volume/${modality}/${run}/head.nii.gz"
    original_mask = "${prepare_dir}/Volume/${modality}/${run}/brainmask.nii.gz"
    fixed_mask = "${prepare_dir}/Volume/${modality}/${run}/brainmask_fixed.nii.gz"
    session_qc_dir = "${qc_dir}/${subj}/${ses}"
    qc_output = "${session_qc_dir}/qc_fixed_brainmask_${modality}_${run}.png"
    if ( params.process_stage == 'all' || params.process_stage == 'prepare' )
      """
      set -euo pipefail
      mkdir -p "${session_qc_dir}"

      # Only run QC if good_mask was false (meaning fix was attempted)
      if [[ "${good_mask}" == "false" ]]; then
        if [[ -f "${head_path}" && -f "${original_mask}" ]]; then
          # Use fixed mask if exists, otherwise use original
          if [[ -f "${fixed_mask}" ]]; then
            MASK_ARGS="--original_mask ${original_mask} --fixed_mask ${fixed_mask}"
          else
            MASK_ARGS="--original_mask ${original_mask} --fixed_mask ${original_mask}"
          fi

          ${params.python_inter} ${params.utils_path}/qc_fixed_brainmask.py \
            --head "${head_path}" \
            \${MASK_ARGS} \
            --output "${qc_output}"
          echo "[OK] QC fixed brainmask saved to ${qc_output}"
        else
          echo "[WARN] Missing files, skip QC for ${run}"
        fi
      else
        echo "[INFO] good_mask=true, skip fixed brainmask QC"
      fi
      """
    else
        """echo skip prepare"""
}


// ---------------------------------------
// 5) Correct orientation (per run)
// ---------------------------------------
process anat_corr_orient {
    tag { "${subj}/${ses}/${modality}/${run}" }
    cpus 1

    input:
      tuple val(run_key), val(subj), val(ses), val(modality),
            val(run), val(is_brain), val(prepare_dir)

    output:
      tuple val(run_key), val(done)

    script:
    done = 1
    if ( params.process_stage == 'all' || params.process_stage == 'prepare' )
      """
      set -euo pipefail
      ${params.python_inter} ${params.prepare_script_dir}/scripts/correct_orient.py \
        --run_name "${run}" \
        --is_brain "${is_brain}" \
        --run_dir "${prepare_dir}/Volume/${modality}/${run}" \
        --refer_path template \
        --python_inter "${params.python_inter}" \
        --utils_path "${params.utils_path}" \
        --t1_template_brain "${params.t1w_template_brain}"
      """
    else
        """echo skip prepare"""
}

// ---------------------------------------
// 5b) Check correct orientation (per run)
// ---------------------------------------
process qc_corr_orient {
    tag { "${subj}/${ses}/${modality}/${run}" }
    cpus 1

    input:
      tuple val(run_key), val(subj), val(ses), val(modality),
            val(run), val(prepare_dir)
      val qc_dir

    output:
      tuple val(run_key), val(qc_done)

    script:
    qc_done = 1
    original_path = "${prepare_dir}/Volume/${modality}/${run}/brain.nii.gz"
    corrected_path = "${prepare_dir}/Volume/${modality}/${run}/brain_reorient.nii.gz"
    session_qc_dir = "${qc_dir}/${subj}/${ses}"
    qc_output = "${session_qc_dir}/qc_corr_orient_${modality}_${run}.png"
    if ( params.process_stage == 'all' || params.process_stage == 'prepare' )
      """
      set -euo pipefail
      mkdir -p "${session_qc_dir}"

      if [[ -f "${corrected_path}" ]]; then
        ORIG_ARG=""
        if [[ -f "${original_path}" ]]; then
          ORIG_ARG="--original ${original_path}"
        fi

        ${params.python_inter} ${params.utils_path}/qc_corr_orient.py \
          --corrected "${corrected_path}" \
          \${ORIG_ARG} \
          --output "${qc_output}"
        echo "[OK] QC corr_orient saved to ${qc_output}"
      else
        echo "[WARN] Missing corrected image, skip QC for ${run}"
      fi
      """
    else
        """echo skip prepare"""
}

// ---------------------------------------
// 6) Alignment (per run, needs same-modality ref in session)
//    - if current run is reference: skip
// ---------------------------------------
process anat_alignment {
    tag { "${subj}/${ses}/${modality}/${run}" }
    cpus 1

    input:
      tuple val(run_key), val(subj), val(ses), val(modality), val(run),
            val(ref_run), val(prepare_dir)

    output:
      tuple val(run_key), val(done)

    script:
    done = 1
    if ( params.process_stage == 'all' || params.process_stage == 'prepare' )
      """
      set -euo pipefail

      MOD="${modality}"
      RUN="${run}"
      REF="${ref_run}"
      PREP="${prepare_dir}"

      echo "[DBG] MOD=\$MOD RUN=\$RUN REF=\$REF"

      if [[ "\$REF" == "\$RUN" ]]; then
        echo "[INFO] \$RUN is reference, skip alignment"
        exit 0
      fi

      # 防御：ref 的 reorient 必须存在（gate 已保证 corr_done，但这里再检查一次更直观）
      if [[ ! -f "\$PREP/Volume/\$MOD/\$REF/brain_reorient.nii.gz" ]]; then
        echo "[ERROR] Missing reference reorient: \$PREP/Volume/\$MOD/\$REF/brain_reorient.nii.gz" >&2
        exit 1
      fi

      ${params.python_inter} ${params.prepare_script_dir}/scripts/register_refer.py \
        --refer_dir "\$PREP/Volume/\$MOD/\$REF" \
        --run_dir   "\$PREP/Volume/\$MOD/\$RUN" \
        --python_inter "${params.python_inter}" \
        --utils_path "${params.utils_path}"
      """
    else
        """echo skip prepare"""
}

// ---------------------------------------
// 7) Average (per session+modality group)
// ---------------------------------------
process anat_average {
    tag { "${subj}/${ses}/${modality}" }
    cpus 1

    input:
      tuple val(subj), val(ses), val(modality),
            val(run_names), val(ref_name),
            val(prepare_dir)

    output:
      tuple val(subj), val(ses), val(modality),
            val(head), val(brain), val(brainmask)

    script:
    bids_suffix = modality == "FLAIR" ? "FLAIR" : "${modality}w"
    head = "${subj}_${ses}_desc-head_${bids_suffix}.nii.gz"
    brain = "${subj}_${ses}_desc-brain_${bids_suffix}.nii.gz"
    brainmask = "${subj}_${ses}_desc-brain_mask_${bids_suffix}.nii.gz"
    def runs_str = run_names.join(' ')

    if ( params.process_stage == 'all' || params.process_stage == 'prepare' )
      """
      set -euo pipefail

      MOD="${modality}"
      PREP="${prepare_dir}"
      REF="${ref_name}"

      echo "[INFO] average ${subj}/${ses}/${modality}"
      echo "[INFO] ref(in) : ${ref_name}"
      echo "[INFO] runs    : ${runs_str}"

      out_brain_wd="${brain}"
      out_head_wd="${head}"
      out_mask_wd="${brainmask}"

      out_brain="${prepare_dir}/${brain}"
      out_head="${prepare_dir}/${head}"
      out_mask="${prepare_dir}/${brainmask}"

      if [[ -z "\$REF" || ! -f "\$PREP/Volume/\$MOD/\$REF/brainmask_reorient.nii.gz" ]]; then
        echo "[WARN] REF invalid or missing mask for \$MOD: '\$REF' -> try pick first run with brainmask_reorient"

        REF=""
        for r in ${runs_str}; do
          if [[ -f "\$PREP/Volume/\$MOD/\$r/brainmask_reorient.nii.gz" ]]; then
            REF="\$r"
            break
          fi
        done

        if [[ -z "\$REF" ]]; then
          echo "[ERROR] No usable reference (brainmask_reorient.nii.gz) found for ${subj}/${ses}/${modality}" >&2
          exit 1
        fi
      fi

      echo "[INFO] ref(use): \$REF"

      brain_list=()
      head_list=()

      for r in ${runs_str}; do
        align_dir="\$PREP/Volume/\$MOD/\$r/\${r}_to_\${REF}"

        b="\$align_dir/final_brain.nii.gz"
        h="\$align_dir/final_head.nii.gz"

        [[ -f "\$b" ]] && brain_list+=("\$b")
        [[ -f "\$h" ]] && head_list+=("\$h")
      done

      ref_brain="\$PREP/Volume/\$MOD/\$REF/brain_reorient.nii.gz"
      ref_head="\$PREP/Volume/\$MOD/\$REF/head_reorient.nii.gz"
      [[ -f "\$ref_brain" ]] && brain_list+=("\$ref_brain")
      [[ -f "\$ref_head"  ]] && head_list+=("\$ref_head")

      if [[ \${#brain_list[@]} -gt 0 ]]; then
        ${params.python_inter} ${params.prepare_script_dir}/scripts/average.py \
          --input_files "\${brain_list[@]}" \
          --output_file "\$out_brain_wd"

        if [[ \${#brain_list[@]} -lt 4 ]]; then
          DenoiseImage -d 3 -i "\$out_brain_wd" -n Rician -o "\$out_brain_wd" -r ${params.denoise_rician_rad}
        fi

        cp -f "\$out_brain_wd" "\$out_brain"
      else
        echo "[ERROR] No brain inputs found for averaging ${subj}/${ses}/${modality} (REF=\$REF)" >&2
        exit 1
      fi

      if [[ \${#head_list[@]} -gt 0 ]]; then
        ${params.python_inter} ${params.prepare_script_dir}/scripts/average.py \
          --input_files "\${head_list[@]}" \
          --output_file "\$out_head_wd"

        if [[ \${#head_list[@]} -lt 5 ]]; then
          DenoiseImage -d 3 -i "\$out_head_wd" -n Rician -o "\$out_head_wd" -r ${params.denoise_rician_rad}
        fi

        cp -f "\$out_head_wd" "\$out_head"
      else
        echo "[WARN] No head inputs; copy brain as head"
        cp -f "\$out_brain_wd" "\$out_head_wd"
        cp -f "\$out_head_wd" "\$out_head"
      fi

      ref_mask="\$PREP/Volume/\$MOD/\$REF/brainmask_reorient.nii.gz"
      if [[ -f "\$ref_mask" ]]; then
        cp -f "\$ref_mask" "\$out_mask_wd"
        cp -f "\$out_mask_wd" "\$out_mask"
      else
        echo "[ERROR] Reference mask missing even after selection: \$ref_mask" >&2
        exit 1
      fi

      echo "[DONE] average ${subj}/${ses}/${modality} (REF=\$REF)"
      """
    else
        """echo skip prepare"""
}


// ---------------------------------------
// Workflow: PREPARE (FULL)
// ---------------------------------------
workflow prepare {
  take:
    qc_dir
    ctx_keyed
    run_ctx_keyed
    run_ctx_by_sess
  
  main:
    /*
    * alignment only: build default ref per (subj,ses,modality)
    * output: (sessmod_key, default_ref_run)
    */
    align_default_ref = run_ctx_by_sess
    .map { session_key, m ->
        def sessmod = "${m.subj}::${m.ses}::${m.modality}"
        tuple(sessmod, m.run)
    }
    .groupTuple()
    .map { sessmod, runs ->
        // 稳定选择：按字典序取第一个（你也可以改成别的规则）
        def pick = (runs as List).collect{ it.toString().trim() }.sort()[0]
        tuple(sessmod, pick)
    }

    // --------------------
    // 5) SKULLSTRIP per run
    // --------------------
    skull_in = run_ctx_keyed.map { run_key, m ->
      tuple(run_key, m.subj, m.ses, m.modality, m.run, m.orig,
            m.is_brain, m.good_mask, m.is_refer, m.good_quality, m.prepare_dir)
    }

    skull_done = anat_skullstrip(skull_in)
    // skull_done: tuple(run_key, done)

    // QC: skullstrip
    qc_skullstrip_in = skull_done
      .join(run_ctx_keyed)
      .map { run_key, _done, m ->
        tuple(run_key, m.subj, m.ses, m.modality, m.run, m.prepare_dir)
      }
    qc_skullstrip_done = qc_skullstrip(qc_skullstrip_in, qc_dir)

    // handy keyed version
    skull_done_keyed = skull_done.map { run_key, done -> tuple(run_key, done) }

    // --------------------
    // 6) GATE: fix must wait for (self skull done) + (chosen reference skull done)
    //    reference selection rule:
    //      same modality ref if exists
    //      else T1 ref, else T2 ref, else FLAIR ref
    //      else ERROR
    // --------------------
    self_ready = skull_done_keyed
      .join(run_ctx_keyed)     // join on run_key
      .map { run_key, _done, m ->
        tuple(run_key, m)
      }

    

    // compute chosen reference per run (may be other modality)
    // produce (ref_run_key, payload) then join with skull_done_keyed to enforce ref skull done
    fix_ready = self_ready
      .map { run_key, m ->
        def mod = m.modality
        def refName = ""
        def refMod  = ""

        // 1) same modality
        if( mod == 'T1' && m.refT1 ) { refName = m.refT1; refMod='T1' }
        else if( mod == 'T2' && m.refT2 ) { refName = m.refT2; refMod='T2' }
        else if( mod == 'FLAIR' && m.refFL ) { refName = m.refFL; refMod='FLAIR' }

        // 2) fallback T1 -> T2 -> FLAIR
        if( !refName ) {
          if( m.refT1 ) { refName = m.refT1; refMod='T1' }
          else if( m.refT2 ) { refName = m.refT2; refMod='T2' }
          else if( m.refFL ) { refName = m.refFL; refMod='FLAIR' }
        }

        // 3) still none -> ERROR
        if( !refName ) {
          throw new IllegalStateException("No reference found in session ${m.subj}/${m.ses} (need at least one of refT1/refT2/refFL).")
        }

        def ref_run_key = "${m.subj}::${m.ses}::${refMod}::${refName}"
        // key by ref_run_key for gating
        tuple(ref_run_key, [run_key:run_key, m:m, ref_run_key:ref_run_key])
      }
      .combine(skull_done_keyed, by: 0)     // join on ref_run_key => wait until reference skull done
      .map { ref_run_key, payload, _ref_done ->
        // gate passed
        def run_key = payload.run_key
        def m = payload.m
        tuple(run_key, m)
      }
    
    // --------------------
    // 7) FIX_BRAINMASK per run (now properly gated)
    // --------------------
    fix_in = fix_ready
      .map { run_key, m ->
        tuple(run_key, m.subj, m.ses, m.modality, m.run,
              m.good_mask, m.refT1, m.refT2, m.refFL, m.prepare_dir)
      }

    fix_done = anat_fix_brainmask(fix_in)   // tuple(run_key, done)

    // QC: fixed brainmask
    qc_fixed_brainmask_in = fix_done
      .join(run_ctx_keyed)
      .map { run_key, _done, m ->
        tuple(run_key, m.subj, m.ses, m.modality, m.run, m.good_mask, m.prepare_dir)
      }
    qc_fixed_brainmask_done = qc_fixed_brainmask(qc_fixed_brainmask_in, qc_dir)

    // --------------------
    // 8) CORRECT ORIENT per run (depends on fix_done)
    // --------------------
    corr_in = fix_done
      .join(run_ctx_keyed)
      .map { run_key, _done, m ->
        tuple(run_key, m.subj, m.ses, m.modality, m.run, m.is_brain, m.prepare_dir)
      }

    corr_done = anat_corr_orient(corr_in)   // tuple(run_key, done)

    // QC: correct orientation
    qc_corr_orient_in = corr_done
      .join(run_ctx_keyed)
      .map { run_key, _done, m ->
        tuple(run_key, m.subj, m.ses, m.modality, m.run, m.prepare_dir)
      }
    qc_corr_orient_done = qc_corr_orient(qc_corr_orient_in, qc_dir)

    // --------------------
    // 9) ALIGNMENT per run (depends on corr_done)
    //    alignment process itself will select ref similarly (same modality else fallback)
    //    but we already ensured chosen-ref skullstrip exists by gating above for fix;
    //    corr_done guarantees this run is ready.
    // --------------------
    /*
    * alignment gate:
    * - 当前 run corr_done 了
    * - 且 reference run 也 corr_done 了（保证 brain_reorient.nii.gz 已产生）
    */

    // corr_done_keyed: (run_key, done)
    corr_done_keyed = corr_done.map { run_key, done -> tuple(run_key, done) }

    // self_corr_ready: (run_key, m)
    self_corr_ready = corr_done_keyed
    .join(run_ctx_keyed)
    .map { run_key, _done, m -> tuple(run_key, m) }

    /*
    * attach (sessmod_key -> default_ref_run),
    * pick same-modality ref only:
    *   - prefer YAML ref for that modality
    *   - else use default_ref_run (same modality)
    * then gate on ref run's corr_done
    */
    align_ready = self_corr_ready
    .map { run_key, m ->
        def sessmod = "${m.subj}::${m.ses}::${m.modality}"
        tuple(sessmod, [run_key:run_key, m:m])
    }
    .combine(align_default_ref, by: 0)  // attach default ref for this modality
    .map { sessmod, payload, default_ref_run ->
        def m = payload.m
        def mod = m.modality

        // 同模态：优先 YAML 指定 ref（只看该模态那一个），否则用 default_ref_run
        def ref_run = ""
        if( mod == 'T1' )    ref_run = (m.refT1 ?: "").toString().trim()
        if( mod == 'T2' )    ref_run = (m.refT2 ?: "").toString().trim()
        if( mod == 'FLAIR' ) ref_run = (m.refFL ?: "").toString().trim()

        if( !ref_run ) ref_run = default_ref_run.toString().trim()

        // 这里 ref_run 一定同模态
        def ref_run_key = "${m.subj}::${m.ses}::${mod}::${ref_run}"

        tuple(ref_run_key, [run_key:payload.run_key, m:m, ref_run:ref_run])
    }
    .combine(corr_done_keyed, by: 0)     // wait reference corr_orient done
    .map { ref_run_key, payload, _ref_done ->
        tuple(payload.run_key, payload.m, payload.ref_run)
    }

    // 最终喂给 anat_alignment：把 ref_run 显式传进去（避免 process 自己乱 fallback）
    align_in = align_ready
    .map { run_key, m, ref_run ->
        tuple(run_key, m.subj, m.ses, m.modality, m.run, ref_run, m.prepare_dir)
    }

    align_done = anat_alignment(align_in)

    // --------------------
    // 10) AVERAGE per session+modality (depends on align_done)
    //     ref_name MUST be the same-modality ref_run chosen in alignment
    // --------------------

    // keep mapping: (run_key, ref_run) from alignment decision
    align_ref_keyed = align_ready.map { run_key, m, ref_run ->
      tuple(run_key, ref_run)
    }

    avg_ready = align_done
      .join(run_ctx_keyed)        // (run_key, done) + (run_key, m)
      .join(align_ref_keyed)      // attach ref_run chosen in alignment
      .map { run_key, _done, m, ref_run ->
        if( !m.good_quality.toString().equalsIgnoreCase('true') ) return null

        def group_id = "${m.subj}::${m.ses}::${m.modality}"   // group by sess+modality
        tuple(group_id, [
          subj: m.subj,
          ses : m.ses,
          modality: m.modality,
          run : m.run,
          ref_run: ref_run?.toString()?.trim(),
          prepare_dir: m.prepare_dir
        ])
      }
      .filter { it != null }

    avg_group = avg_ready
      .groupTuple()
      .map { group_id, rows ->
        def subj = rows[0].subj
        def ses  = rows[0].ses
        def modality = rows[0].modality
        def prepare_dir = rows[0].prepare_dir

        def run_names = rows.collect { it.run }

        // alignment 选出的 ref_run：理论上组内应该一致；不一致则报错（避免 silent wrong）
        def ref_set = rows.collect{ it.ref_run }.findAll{ it && it.size() > 0 }.unique()
        if( ref_set.size() == 0 ) {
          throw new IllegalStateException("Average: missing ref_run from alignment for ${subj}/${ses}/${modality}")
        }
        if( ref_set.size() > 1 ) {
          throw new IllegalStateException("Average: inconsistent ref_run within group ${subj}/${ses}/${modality}: ${ref_set}")
        }

        def ref_name = ref_set[0]
        tuple(subj, ses, modality, run_names, ref_name, prepare_dir)
      }

    avg_out = anat_average(avg_group)

    avg_out = avg_out.map { subj, ses, modality, head, brain, mask -> 
      def session_key = "${subj}::${ses}"
      tuple(session_key, subj, ses, modality, head, brain, mask) 
    }.combine(ctx_keyed, by: 0).map { session_key, subj, ses, modality, head, brain, mask, m -> 
      tuple(session_key, subj, ses, modality, m.prepare_dir, m.enhance_dir, m.surface_dir, m.resample_dir, head, brain, mask)
    }


  emit:
    avg_out
}




process anat_conform {
    tag { "${ses_key}::${modality}" }
    cpus 1

    input:
    tuple val(ses_key), val(subj), val(ses), val(modality),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(avg_head), val(avg_brain), val(avg_mask)
          

    output:
    tuple val(ses_key), val(subj), val(ses), val(modality),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir)
        
    script:
    modal_path = "${enhance_dir}/${modality}w"

    if ( params.process_stage == 'all' || params.process_stage == 'enhance' )
      """
      bash ${params.enhance_script_dir}/scripts/Conform.sh \
          --modality ${modality} \
          --subj ${subj} \
          --ses ${ses} \
          --prepare_dir ${prepare_dir} \
          --enhance_dir ${enhance_dir} \
          --python_inter ${params.python_inter} \
          --utils_path ${params.utils_path}
      """
    else
        """echo skip enhance"""
}

process anat_modality_register {
    tag { "${ses_key}" }
    cpus 2
    
    input:
    tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir)
          
    output:
    tuple val(ses_key),
          val(t1w_conform), val(t2w_conform), val(flairw_conform)
    
    script:
    def (subj, ses) = ses_key.split('::')
    if (modality_list.contains('T2')) contain_t2 = 'True' else contain_t2 = 'False'
    if (modality_list.contains('FLAIR')) contain_flair = 'True' else contain_flair = 'False'
    t1w_path = "${enhance_dir[0]}/T1w"
    t1w_conform = "${t1w_path}/${subj}_${ses}_desc-conform_T1w.nii.gz"
    t2w_conform = "${t1w_path}/${subj}_${ses}_desc-conform_T2w.nii.gz"
    flairw_conform = "${t1w_path}/${subj}_${ses}_desc-conform_FLAIR.nii.gz"

    if ( params.process_stage == 'all' || params.process_stage == 'enhance' )
      """
      bash ${params.enhance_script_dir}/scripts/ModalityReg.sh \
          --contain_t2 "${contain_t2}" \
          --contain_flair "${contain_flair}" \
          --enhance_dir ${enhance_dir[0]} \
          --device ${params.device} \
          --script_dir ${params.enhance_script_dir}
      """
    else
        """echo skip enhance"""
}

process qc_modality_register {
  tag { "${ses_key}" }
  cpus 1

  input:
    tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(t1w_conform), val(t2w_conform), val(flairw_conform)
    val qc_dir

  output:
    tuple val(ses_key), val(qc_done)

  script:
    qc_done = 1
    def (subj, ses) = ses_key.split('::')
    session_qc_dir = "${qc_dir}/${subj}/${ses}"
    qc_output = "${session_qc_dir}/qc_modality_register.png"
    if ( params.process_stage == 'all' || params.process_stage == 'enhance' )
      """
      set -euo pipefail
      mkdir -p "${session_qc_dir}"

      T1W_ARG="--target ${t1w_conform}"
      T2W_ARG=""
      FLAIR_ARG=""

      if [[ -f "${t2w_conform}" ]]; then
        T2W_ARG="--t2w ${t2w_conform}"
      fi

      if [[ -f "${flairw_conform}" ]]; then
        FLAIR_ARG="--flair ${flairw_conform}"
      fi

      ${params.python_inter} ${params.utils_path}/qc_modality_register.py \
        \${T1W_ARG} \
        \${T2W_ARG} \
        \${FLAIR_ARG} \
        --output "${qc_output}"
      echo "[OK] QC modality_register saved to ${qc_output}"
      """
    else
          """echo skip enhance"""
}

process anat_init_template_register {
    tag { "${ses_key}" }
    accelerator 1
    maxForks 1 * ((params.per_gpu == null) ? 1 : (params.per_gpu as int))

    input:
    tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(t1w_conform), val(t2w_conform), val(flairw_conform)

    output:
    tuple val(ses_key),
          val(t1w_init_corrected), val(t1w_aseg),
          val(t1w_wm_compliment1), val(t1w_wm_compliment2), val(t1w_gm_compliment),
          val(t1w_restore_brain)
    

    script:
    def (subj, ses) = ses_key.split('::')
    t1w_space_path = "${enhance_dir[0]}/T1w"
    template_space_path = "${enhance_dir[0]}/MEBRAIN"
    t1w_init_corrected = "${t1w_space_path}/${subj}_${ses}_desc-initcorrected_T1w.nii.gz"
    t1w_aseg = "${t1w_space_path}/${subj}_${ses}_desc-aseg_dseg.nii.gz"
    t1w_wm_compliment1 = "${t1w_space_path}/${subj}_${ses}_desc-wmcomp1_mask.nii.gz"
    t1w_wm_compliment2 = "${t1w_space_path}/${subj}_${ses}_desc-wmcomp2_mask.nii.gz"
    t1w_gm_compliment = "${t1w_space_path}/${subj}_${ses}_desc-gmcomp_mask.nii.gz"
    t1w_restore_brain = "${template_space_path}/${subj}_${ses}_space-MEBRAIN_desc-restorebrain_T1w.nii.gz"
    if ( params.process_stage == 'all' || params.process_stage == 'enhance' )
      """
      bash ${params.enhance_script_dir}/scripts/TemplateReg.sh \
          --t1w_conform ${t1w_conform} \
          --t2w_conform ${t2w_conform} \
          --enhance_dir ${enhance_dir[0]} \
          --device ${params.device} \
          --script_dir ${params.enhance_script_dir} \
          --t1w_template ${params.t1w_template} \
          --t1w_template_brain ${params.t1w_template_brain} \
          --template_mask ${params.template_mask} \
          --t1w_template_2mm ${params.t1w_template_2mm} \
          --template_2mm_mask ${params.template_2mm_mask} \
          --fnirt_config ${params.fnirt_config} \
          --t1w_template_atlas ${params.t1w_template_atlas} \
          --wm_compliment1 ${params.wm_compliment1} \
          --wm_compliment2 ${params.wm_compliment2} \
          --gm_compliment ${params.gm_compliment}
      """
    else
        """echo skip enhance"""
}

process qc_template_register {
  tag { "${ses_key}" }
  cpus 1

  input:
    tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(t1w_init_corrected), val(t1w_aseg),
          val(t1w_wm_compliment1), val(t1w_wm_compliment2), val(t1w_gm_compliment), val(t1w_restore_brain)
    val qc_dir

  output:
    tuple val(ses_key), val(qc_done)

  script:
  qc_done = 1
  def (subj, ses) = ses_key.split('::')
  session_qc_dir = "${qc_dir}/${subj}/${ses}"
  qc_output = "${session_qc_dir}/qc_template_register.png"
  if ( params.process_stage == 'all' || params.process_stage == 'enhance' )
    """
    set -euo pipefail
    mkdir -p "${session_qc_dir}"

    ASEG_ARG=""
    if [[ -f "${t1w_aseg}" ]]; then
      ASEG_ARG="--aseg ${t1w_aseg}"
    fi

    ${params.python_inter} ${params.utils_path}/qc_template_register.py \
      --subject-orig "${t1w_init_corrected}" \
      --subject-reg "${t1w_restore_brain}" \
      --template "${params.t1w_template_brain}" \
      \${ASEG_ARG} \
      --output "${qc_output}"
    echo "[OK] QC template_register saved to ${qc_output}"
    """
  else
        """echo skip enhance"""
}

process anat_tissue_segment {
    tag { "${ses_key}" }
    accelerator 1
    maxForks 1 * ((params.per_gpu == null) ? 1 : (params.per_gpu as int))

    input:
    tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(t1w_init_corrected), val(t1w_aseg),
          val(t1w_wm_compliment1), val(t1w_wm_compliment2), val(t1w_gm_compliment), val(t1w_restore_brain)

    output:
    tuple val(ses_key),
          val(t1w_cerebellum_brainstem), val(t1w_cerebrum), val(t1w_nbest),
          val(t1w_complete_aseg)

    script:
    def (subj, ses) = ses_key.split('::')
    t1w_path = "${enhance_dir[0]}/T1w"
    t1w_cerebellum_brainstem = "${t1w_path}/${subj}_${ses}_label-cerebellum-brainstem_dseg.nii.gz"
    t1w_cerebrum = "${t1w_path}/${subj}_${ses}_label-cerebrum_dseg.nii.gz"
    t1w_nbest = "${t1w_path}/${subj}_${ses}_desc-nbest_dseg.nii.gz"
    t1w_complete_aseg = "${t1w_path}/${subj}_${ses}_desc-completeaseg_dseg.nii.gz"
    t1w_conform_mask = "${t1w_path}/${subj}_${ses}_desc-conform_mask.nii.gz"
    predict_script = "${params.maca_brainnet_dir}/predict_ensemble.py"
    tissue_model_dir = "${params.maca_brainnet_dir}/swinunetr_models/tissue_segmentation"
    if ( params.process_stage == 'all' || params.process_stage == 'enhance' )
      """
      set -euo pipefail
      if [ "${params.seg_tool}" == "nbest" ]; then
          ${params.python_inter} ${params.enhance_script_dir}/scripts/nbest_tissueseg.py \\
              --input ${t1w_init_corrected} \\
              --mask ${t1w_conform_mask} \\
              --template-aseg ${t1w_aseg} \\
              --output-dir ${t1w_path} \\
              --prefix ${subj}_${ses} \\
              --nbest-model-path ${params.nbest_model_path} \\
              --python-env ${params.python_env} \\
              --python-inter ${params.python_inter} \\
              --utils-path ${params.utils_path}
      else
          ${params.python_inter} ${params.enhance_script_dir}/scripts/brainnet_tissueseg.py \\
              --input ${t1w_init_corrected} \\
              --output-dir ${t1w_path} \\
              --prefix ${subj}_${ses} \\
              --model-dir ${tissue_model_dir} \\
              --predict-script ${predict_script} \\
              --python-inter ${params.python_inter} \\
              --utils-path ${params.utils_path}
      fi
      """
    else
        """echo skip enhance"""
}

process qc_tissue_segment {
    tag { "${ses_key}" }
    cpus 1

    input:
      tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
            val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
            val(t1w_cerebellum_brainstem), val(t1w_cerebrum), val(t1w_nbest),
            val(t1w_complete_aseg), val(t1w_conform), val(t2w_conform), val(flairw_conform)
      val qc_dir

    output:
      tuple val(ses_key), val(qc_done)

    script:
    qc_done = 1
    def (subj, ses) = ses_key.split('::')
    session_qc_dir = "${qc_dir}/${subj}/${ses}"
    qc_output = "${session_qc_dir}/qc_tissue_segment.png"
    if ( params.process_stage == 'all' || params.process_stage == 'enhance' )
      """
      set -euo pipefail
      mkdir -p "${session_qc_dir}"

      NBEST_ARG=""
      CEREB_ARG=""
      CEREBRUM_ARG=""

      if [[ -f "${t1w_nbest}" ]]; then
        NBEST_ARG="--nbest ${t1w_nbest}"
      fi

      if [[ -f "${t1w_cerebellum_brainstem}" ]]; then
        CEREB_ARG="--cerebellum-brainstem ${t1w_cerebellum_brainstem}"
      fi

      if [[ -f "${t1w_cerebrum}" ]]; then
        CEREBRUM_ARG="--cerebrum ${t1w_cerebrum}"
      fi

      ${params.python_inter} ${params.utils_path}/qc_tissue_segment.py \
        --t1w "${t1w_conform}" \
        \${NBEST_ARG} \
        \${CEREB_ARG} \
        \${CEREBRUM_ARG} \
        --output "${qc_output}"
      echo "[OK] QC tissue_segment saved to ${qc_output}"
      """
    else
        """echo skip enhance"""
}

process anat_bias_field_corr {
    tag { "${ses_key}" }
    cpus 2
    memory { 1.5.GB * task.attempt }

    input:
    tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(t1w_cerebellum_brainstem), val(t1w_cerebrum), val(t1w_nbest),
          val(t1w_complete_aseg), val(t1w_conform), val(t2w_conform), val(flairw_conform)

    output:
    tuple val(ses_key),
          val(t1w_final_corrected), val(t2w_final_corrected),
          val(t1w_white), val(t1w_pial), val(t2w_pial)

    script:
    def (subj, ses) = ses_key.split('::')
    if (modality_list.contains('T2')) contain_t2 = 'True' else contain_t2 = 'False'
    if (modality_list.contains('FLAIR')) contain_flair = 'True' else contain_flair = 'False'
    t1w_path = "${enhance_dir[0]}/T1w"
    t1w_final_corrected = "${t1w_path}/${subj}_${ses}_desc-bfc_T1w.nii.gz"
    t2w_final_corrected = "${t1w_path}/${subj}_${ses}_desc-bfc_T2w.nii.gz"
    t1w_white = "${t1w_path}/${subj}_${ses}_desc-whitebfc_T1w.nii.gz"
    t1w_pial = "${t1w_path}/${subj}_${ses}_desc-pialbfc_T1w.nii.gz"
    t2w_pial = "${t1w_path}/${subj}_${ses}_desc-pialbfc_T2w.nii.gz"

    if ( params.process_stage == 'all' || params.process_stage == 'enhance' || params.process_stage == 'biasfield' )
      """
      t1w_complete_aseg_merged="${t1w_path}/${subj}_${ses}_desc-completeaseg-merged_dseg.nii.gz"
      tissue19_args="--t1w_tissue19 \${t1w_complete_aseg_merged} --use_tissue19 true"
      echo "[INFO] Using merged complete aseg for BFC"

      bash ${params.enhance_script_dir}/scripts/BiasFieldCorrect.sh \
          --t1w_conform ${t1w_conform} \
          --t2w_conform ${t2w_conform} \
          --t1w_nbest ${t1w_nbest} \
          --t1w_cerebellum_brainstem ${t1w_cerebellum_brainstem} \
          --enhance_dir ${enhance_dir[0]} \
          --contain_t2 ${contain_t2} \
          --bfc_method ${params.bfc_method} \
          --python_inter ${params.python_inter} \
          --utils_path ${params.utils_path} \
          --script_dir ${params.enhance_script_dir} \
          \${tissue19_args}
      """
    else
        """echo skip enhance"""
}

process qc_bias_field_corr {
  tag { "${ses_key}" }
  cpus 1

  input:
    tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(t1w_cerebellum_brainstem), val(t1w_cerebrum), val(t1w_nbest),
          val(t1w_complete_aseg), val(t1w_conform), val(t2w_conform), val(flairw_conform),
          val(t1w_final_corrected), val(t2w_final_corrected),
          val(t1w_white), val(t1w_pial), val(t2w_pial)
    val qc_dir

  output:
    tuple val(ses_key), val(qc_done)

  script:
  qc_done = 1
  def (subj, ses) = ses_key.split('::')
  session_qc_dir = "${qc_dir}/${subj}/${ses}"
  t1w_path = "${enhance_dir[0]}/T1w"
  original_path = "${t1w_path}/${subj}_${ses}_desc-conform_T1w.nii.gz"
  corrected_path = "${t1w_path}/${subj}_${ses}_desc-bfc_T1w.nii.gz"
  white_path = "${t1w_path}/${subj}_${ses}_desc-whitebfc_T1w.nii.gz"
  pial_path = "${t1w_path}/${subj}_${ses}_desc-pialbfc_T1w.nii.gz"
  qc_output = "${session_qc_dir}/qc_bias_field_corr.png"

  if ( params.process_stage == 'all' || params.process_stage == 'enhance' || params.process_stage == 'biasfield' )
    """
    set -euo pipefail
    mkdir -p "${session_qc_dir}"

    ORIG_ARG=""
    WHITE_ARG=""
    PIAL_ARG=""

    if [[ -f "${original_path}" ]]; then
      ORIG_ARG="--original ${original_path}"
    fi

    if [[ -f "${white_path}" ]]; then
      WHITE_ARG="--white ${white_path}"
    fi

    if [[ -f "${pial_path}" ]]; then
      PIAL_ARG="--pial ${pial_path}"
    fi

    ${params.python_inter} ${params.utils_path}/qc_bias_field_corr.py \
      --corrected "${corrected_path}" \
      \${ORIG_ARG} \
      \${WHITE_ARG} \
      \${PIAL_ARG} \
      --output "${qc_output}"
    echo "[OK] QC bias_field_corr saved to ${qc_output}"
    """
  else
        """echo skip enhance"""
}

process anat_detect_vessel {
    tag { "${ses_key}" }
    cpus 1

    input:
    tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(t1w_cerebellum_brainstem), val(t1w_cerebrum), val(t1w_nbest),
          val(t1w_complete_aseg),
          val(t1w_init_corrected), val(t1w_aseg),
          val(t1w_wm_compliment1), val(t1w_wm_compliment2), val(t1w_gm_compliment), val(t1w_restore_brain),
          val(t1w_final_corrected), val(t2w_final_corrected),
          val(t1w_white), val(t1w_pial), val(t2w_pial)


    output:
    tuple val(ses_key),
          val(t1w_vessel)

    script:
    def (subj, ses) = ses_key.split('::')
    if (modality_list.contains('T2')) contain_t2 = 'True' else contain_t2 = 'False'
    if (modality_list.contains('FLAIR')) contain_flair = 'True' else contain_flair = 'False'
    t1w_path = "${enhance_dir[0]}/T1w"
    t1w_vessel = "${t1w_path}/${subj}_${ses}_desc-bfc_label-vessel_T1w.nii.gz"

    if ( params.process_stage == 'all' || params.process_stage == 'enhance' || params.process_stage == 'detect_vessel' )
      """
      bash ${params.enhance_script_dir}/scripts/VesselDetect.sh \
          --t1w_cerebellum_brainstem ${t1w_cerebellum_brainstem} \
          --t1w_aseg ${t1w_aseg} \
          --t1w_nbest ${t1w_nbest} \
          --t1w_final_corrected ${t1w_final_corrected} \
          --t2w_final_corrected ${t2w_final_corrected} \
          --enhance_dir ${enhance_dir[0]} \
          --contain_t2 ${contain_t2} \
          --python_inter ${params.python_inter} \
          --utils_path ${params.utils_path} \
          --vessel_detect ${params.vessel_detect}
      """
    else
        """echo skip enhance"""
}

process anat_fake_t2 {
    tag { "${ses_key}" }
    cpus 1

    input:
    tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(t1w_final_corrected), val(t2w_final_corrected),
          val(t1w_white), val(t1w_pial), val(t2w_pial),
          val(t1w_vessel)

    output:
    tuple val(ses_key),
          val(t2w_final_corrected), val(t2w_pial), val(t2w_vessel)

    script:
    def (subj, ses) = ses_key.split('::')
    if (modality_list.contains('T2')) contain_t2 = 'True' else contain_t2 = 'False'
    if (modality_list.contains('FLAIR')) contain_flair = 'True' else contain_flair = 'False'
    t1w_path = "${enhance_dir[0]}/T1w"
    t2w_final_corrected = "${t1w_path}/${subj}_${ses}_desc-bfc_T2w.nii.gz"
    t2w_pial = "${t1w_path}/${subj}_${ses}_desc-pialbfc_T2w.nii.gz"
    t2w_vessel = "${t1w_path}/${subj}_${ses}_desc-bfc_label-vessel_T2w.nii.gz"

    if ( params.process_stage == 'all' || params.process_stage == 'enhance' || params.process_stage == 'fake_t2' )
      """
      bash ${params.enhance_script_dir}/scripts/T2wFake.sh \
          --t1w_final_corrected ${t1w_final_corrected} \
          --t1w_pial ${t1w_pial} \
          --t1w_vessel ${t1w_vessel} \
          --enhance_dir ${enhance_dir[0]} \
          --contain_t2 ${contain_t2}
      """
    else
        """echo skip enhance"""
}

process anat_acpc_isotropy {
    tag { "${ses_key}" }
    cpus 1

    input:
    tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(t1w_conform), val(t2w_conform), val(flairw_conform),
          val(t1w_init_corrected), val(t1w_aseg),
          val(t1w_wm_compliment1), val(t1w_wm_compliment2), val(t1w_gm_compliment), val(t1w_restore_brain),
          val(t1w_cerebellum_brainstem), val(t1w_cerebrum), val(t1w_nbest),
          val(t1w_complete_aseg),
          val(t1w_final_corrected), val(t2w_final_corrected),
          val(t1w_white), val(t1w_pial), val(t2w_pial),
          val(t1w_vessel),
          val(fake_t2w_final_corrected), val(fake_t2w_pial), val(t2w_vessel)

    output:
    tuple val(ses_key),
          val(t1w_acpc_iso), val(t2w_acpc_iso), val(acpc_mat),
          val(t1w_acpc_complete_aseg)

    script:
    def (subj, ses) = ses_key.split('::')
    if (modality_list.contains('T2')) contain_t2 = 'True' else contain_t2 = 'False'
    if (modality_list.contains('FLAIR')) contain_flair = 'True' else contain_flair = 'False'
    t1w_path = "${enhance_dir[0]}/T1w"
    t1w_acpc_iso = "${t1w_path}/${subj}_${ses}_space-acpc_res-04mm_desc-brain_T1w.nii.gz"
    t2w_acpc_iso = "${t1w_path}/${subj}_${ses}_space-acpc_res-04mm_desc-brain_T2w.nii.gz"
    acpc_mat = "${t1w_path}/xfms/acpc.mat"
    t1w_acpc_complete_aseg = "${t1w_path}/${subj}_${ses}_space-acpc_res-04mm_desc-completeaseg_dseg.nii.gz"

    if ( params.process_stage == 'all' || params.process_stage == 'enhance' || params.process_stage == 'acpc_isotropy' )
      """
      bash ${params.enhance_script_dir}/scripts/AcpcIsotropy.sh \
          --t1w_conform ${t1w_conform} \
          --t1w_cerebrum ${t1w_cerebrum} \
          --t1w_aseg ${t1w_aseg} \
          --t1w_nbest ${t1w_nbest} \
          --t1w_complete_aseg ${t1w_complete_aseg} \
          --t1w_final_corrected ${t1w_final_corrected} \
          --t1w_white ${t1w_white} \
          --t1w_pial ${t1w_pial} \
          --t1w_vessel ${t1w_vessel} \
          --t2w_final_corrected ${contain_t2 == 'True' ? t2w_final_corrected : fake_t2w_final_corrected} \
          --t2w_pial ${contain_t2 == 'True' ? t2w_pial : fake_t2w_pial} \
          --t2w_vessel ${t2w_vessel} \
          --wm_comp1 ${t1w_wm_compliment1} \
          --wm_comp2 ${t1w_wm_compliment2} \
          --gm_comp ${t1w_gm_compliment} \
          --enhance_dir ${enhance_dir[0]} \
          --contain_t2 ${contain_t2} \
          --python_inter ${params.python_inter} \
          --utils_path ${params.utils_path} \
          --t1w_template ${params.t1w_template} \
          --t1w_template_brain ${params.t1w_template_brain}
      """
    else
        """echo skip enhance"""
}

process anat_fix_wm {
    tag { "${ses_key}" }
    cpus 2
    memory { 500.MB * task.attempt }

    input:
    tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(t1w_init_corrected), val(t1w_aseg),
          val(t1w_wm_compliment1), val(t1w_wm_compliment2), val(t1w_gm_compliment), val(t1w_restore_brain),
          val(t1w_cerebellum_brainstem), val(t1w_cerebrum), val(t1w_nbest),
          val(t1w_complete_aseg),
          val(t1w_acpc_iso), val(t2w_acpc_iso), val(acpc_mat),
          val(t1w_acpc_complete_aseg)

    output:
    tuple val(ses_key),
          val(t1w_white_skeleton)

    script:
    def (subj, ses) = ses_key.split('::')
    t1w_path = "${enhance_dir[0]}/T1w"
    workdir = "${t1w_path}/fix"
    t1w_white_skeleton = "${t1w_path}/${subj}_${ses}_space-acpc_res-04mm_desc-wmskeleton_mask.nii.gz"

    if ( params.process_stage == 'all' || params.process_stage == 'enhance' || params.process_stage == 'fix_wm' )
      """
      bash ${params.enhance_script_dir}/scripts/WhiteFix.sh \
          --t1w_nbest ${t1w_nbest} \
          --t1w_aseg ${t1w_aseg} \
          --t1w_complete_aseg ${t1w_acpc_complete_aseg} \
          --enhance_dir ${enhance_dir[0]} \
          --atlas_folder ${params.gca_dir} \
          --device ${params.device} \
          --python_inter ${params.python_inter} \
          --utils_path ${params.utils_path} \
          --fix_white ${params.fix_white}
      """
    else
        """echo skip enhance"""
}

workflow enhance {
    take:
    qc_dir
    ses_data_in

    main:
    ses_conform_out = anat_conform(ses_data_in)
    ses_conform_out = ses_conform_out.groupTuple(by: 0) // base info of session

    ses_modality_register_in = ses_conform_out
    ses_modality_register_out = anat_modality_register(ses_modality_register_in) // t1w_conform, t2w_conform, flairw_conform

    // QC: modality register
    qc_modality_register_in = ses_conform_out.join(ses_modality_register_out)
    qc_modality_register_done = qc_modality_register(qc_modality_register_in, qc_dir)

    ses_init_template_register_in = ses_conform_out.join(ses_modality_register_out)
    ses_init_template_register_out = anat_init_template_register(ses_init_template_register_in) // t1w_init_corrected, t1w_aseg, t1w_wm_compliment1, t1w_wm_compliment2, t1w_gm_compliment, t1w_restore_brain

    // QC: template register
    qc_template_register_in = ses_conform_out.join(ses_init_template_register_out)
    qc_template_register_done = qc_template_register(qc_template_register_in, qc_dir)

    ses_tissue_segment_in = ses_conform_out.join(ses_init_template_register_out)
    ses_tissue_segment_out = anat_tissue_segment(ses_tissue_segment_in) // t1w_cerebellum_brainstem, t1w_cerebrum, t1w_nbest

    // QC: tissue segment
    qc_tissue_segment_in = ses_conform_out.join(ses_tissue_segment_out).join(ses_modality_register_out)
    qc_tissue_segment_done = qc_tissue_segment(qc_tissue_segment_in, qc_dir)

    ses_bias_field_corr_in = ses_conform_out.join(ses_tissue_segment_out).join(ses_modality_register_out)
    ses_bias_field_corr_out = anat_bias_field_corr(ses_bias_field_corr_in) // t1w_final_corrected, t2w_final_corrected, t1w_white, t1w_pial, t2w_pial

    // QC: bias field correction
    qc_bias_field_corr_in = ses_conform_out.join(ses_tissue_segment_out).join(ses_modality_register_out).join(ses_bias_field_corr_out)
    qc_bias_field_corr_done = qc_bias_field_corr(qc_bias_field_corr_in, qc_dir)

    ses_detect_vessel_in = ses_conform_out.join(ses_tissue_segment_out).join(ses_init_template_register_out).join(ses_bias_field_corr_out)
    ses_detect_vessel_out = anat_detect_vessel(ses_detect_vessel_in) // t1w_vessel

    ses_fake_t2_in = ses_conform_out.join(ses_bias_field_corr_out).join(ses_detect_vessel_out)
    ses_fake_t2_out = anat_fake_t2(ses_fake_t2_in) // t2w_final_corrected, t2w_pial, t2w_vessel
    
    ses_acpc_isotropy_in = ses_conform_out.join(ses_modality_register_out).join(ses_init_template_register_out).join(ses_tissue_segment_out).join(ses_bias_field_corr_out).join(ses_detect_vessel_out).join(ses_fake_t2_out)
    ses_acpc_isotropy_out = anat_acpc_isotropy(ses_acpc_isotropy_in) // t1w_acpc_iso, t2w_acpc_iso, acpc_mat

    ses_fix_wm_in = ses_conform_out.join(ses_init_template_register_out).join(ses_tissue_segment_out).join(ses_acpc_isotropy_out)
    ses_fix_wm_out = anat_fix_wm(ses_fix_wm_in) // t1w_white_skeleton

    enhance_out = ses_conform_out.join(ses_acpc_isotropy_out).join(ses_fix_wm_out)

    emit:
    enhance_out
}

process pretess {
    tag { "${ses_key}" }
    cpus 1

    input:
    tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(t1w_acpc_iso), val(t2w_acpc_iso), val(acpc_mat),
          val(t1w_acpc_complete_aseg), val(t1w_white_skeleton)

    output:
    tuple val(ses_key),
          val(filled), val(aseg)

    script:
    filled = "${surface_dir[0]}/workspace/mri/filled.mgz"
    aseg = "${surface_dir[0]}/workspace/mri/aseg.mgz"

    if ( params.process_stage == 'all' || params.process_stage == 'surface' || params.process_stage == 'tessel' )
      """
      bash ${params.surface_script_dir}/scripts/TessellationPre.sh \
          --subject_dir ${surface_dir[0]} \
          --subject_id workspace \
          --t1w_image_file ${t1w_acpc_iso} \
          --num_cores ${params.tessellation_cores} \
          --preprocess_dir ${enhance_dir[0]} \
          --complete_aseg ${t1w_acpc_complete_aseg} \
          --fake_talairch_transform ${params.fake_talairch_transform} \
          --utils_path ${params.utils_path} \
          --python_inter ${params.python_inter} \
          --pipeline_scripts ${params.surface_script_dir}
      """
    else
        """echo skip surface"""
}

process tessellate {
    tag { "${ses_key}::${hemi}" }
    cpus 2

    input:
    tuple val(ses_key), val(hemi), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(filled), val(aseg)

    output:
    tuple val(ses_key), val(hemi),
          val(white_prehires)

    script:
    white_prehires = "${surface_dir[0]}/workspace/surf/${hemi}.white.prehires"

    if ( params.process_stage == 'all' || params.process_stage == 'surface' || params.process_stage == 'tessel' )
      """
      bash ${params.surface_script_dir}/scripts/Tessellation.sh \
          --subject_dir ${surface_dir[0]} \
          --subject_id workspace \
          --hemi ${hemi}
      """
    else
        """echo skip surface"""
}

process prehires {
    tag { "${ses_key}" }
    cpus 1
    
    input:
    tuple val(ses_key), val(hemi_list) ,val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(white_prehires)
    
    output:
    tuple val(ses_key), 
          val(t1w_hires_white), val(t1w_hires_pial_one), val(t1w_hires_pial_two), val(t2w_hires_pial)
    
    script:
    subject_id = "workspace"
    mri_dir = "${surface_dir[0][0]}/${subject_id}/mri"
    t1w_hires_white = "${mri_dir}/T1w_hires_white.nii.gz"
    t1w_hires_pial_one = "${mri_dir}/T1w_hires.norm.one.mgz"
    t1w_hires_pial_two = "${mri_dir}/T1w_hires.norm.two.mgz"
    t2w_hires_pial = "${mri_dir}/T2w_hires.norm.mgz"

    if ( params.process_stage == 'all' || params.process_stage == 'surface' || params.process_stage == 'white' )
      """
      #!/bin/bash
      set -e
      set -x
      
      # 执行高分辨率预处理脚本
      bash ${params.surface_script_dir}/scripts/HiresPre.sh \
          --subject_dir "${surface_dir[0][0]}" \
          --subject_id "${subject_id}" \
          --preprocess_dir "${enhance_dir[0][0]}" \
          --deep_white "${params.deep_white}" \
          --caret7_dir "${params.caret7dir}" \
          --utils_path "${params.utils_path}" \
          --python_inter "${params.python_inter}"
      """
    else
        """echo skip surface"""
}

process whitesurf {
    tag { "${ses_key}::${hemi}" }
    cpus 2

    input:
    tuple val(ses_key), val(hemi), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(white_prehires),
          val(t1w_hires_white), val(t1w_hires_pial_one), val(t1w_hires_pial_two), val(t2w_hires_pial)
    
    output:
    tuple val(ses_key), val(hemi),
          val(white_hires)
    
    script:
    subject_id = "workspace"
    mri_dir = "${surface_dir[0]}/${subject_id}/mri"
    surf_dir = "${surface_dir[0]}/${subject_id}/surf"
    t1w_image = "${mri_dir}/T1w_hires.nii.gz"
    t2w_image = "${mri_dir}/T2w_hires.nii.gz"
    white_hires = "${surf_dir}/${hemi}.white"

    if ( params.process_stage == 'all' || params.process_stage == 'surface' || params.process_stage == 'white' )
      """
      #!/bin/bash
      bash ${params.surface_script_dir}/scripts/HiresWhite.sh \
          --subject_dir "${surface_dir[0]}" \
          --subject_id "${subject_id}" \
          --t1w_image_file "${t1w_image}" \
          --t2w_image_file "${t2w_image}" \
          --hemi "${hemi}" \
          --deep_white "${params.deep_white}" \
          --pipeline_scripts "${params.surface_script_dir}/utils" \
          --utils_path "${params.utils_path}" \
          --python_inter "${params.python_inter}" \
          --species "${params.species}"
      """
    else
        """echo skip surface"""
}

process prereg{
    tag { "${ses_key}" }
    cpus 1

    input:
    tuple val(ses_key), val(hemi_list), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(white_hires)
    
    output:
    tuple val(ses_key), 
          val(lh_medialwall), val(rh_medialwall)
    
    script:
    subject_id = "workspace"
    lh_medialwall = "${surface_dir[0][0]}/${subject_id}/mri/middle/lh_medial_wall_binary.shape.gii"
    rh_medialwall = "${surface_dir[0][0]}/${subject_id}/mri/middle/rh_medial_wall_binary.shape.gii"

    if ( params.process_stage == 'all' || params.process_stage == 'surface' )
      """
      #!/bin/bash
      bash ${params.surface_script_dir}/scripts/RegPre.sh \
          --subject_dir "${surface_dir[0][0]}" \
          --subject_id "${subject_id}" \
          --utils_path "${params.utils_path}" \
          --python_inter "${params.python_inter}"
      """
    else
        """echo skip surface"""
}

process surfreg {
    tag { "${ses_key}::${hemi}" }
    cpus 2
    
    input:
    tuple val(ses_key), val(hemi), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(lh_medialwall), val(rh_medialwall)
    
    output:
    tuple val(ses_key), val(hemi),
          val(fs_sphere_reg)
    script:
    subject_id = "workspace"
    surf_dir = "${surface_dir[0]}/${subject_id}/surf"
    fs_sphere_reg = "${surf_dir}/${hemi}.sphere.reg"

    if ( params.process_stage == 'all' || params.process_stage == 'surface' )
      """
      #!/bin/bash
      bash ${params.surface_script_dir}/scripts/SurfReg.sh \
          --subject_dir "${surface_dir[0]}" \
          --subject_id "${subject_id}" \
          --hemi ${hemi} \
          --utils_path "${params.utils_path}" \
          --python_inter "${params.python_inter}" \
          --gca_dir ${params.gca_dir}
      """
    else
        """echo skip surface"""
}

process msmreg {
    tag { "${ses_key}::${hemi}" }
    cpus 2
    
    input:
    tuple val(ses_key), val(hemi), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(fs_sphere_reg)

    output:
    tuple val(ses_key), val(hemi),
          val(msm_sphere_reg)
    
    script:
    subject_id = "workspace"
    surf_dir = "${surface_dir[0]}/${subject_id}/surf"
    hemi_short = hemi == 'lh' ? 'L' : 'R'
    msm_sphere_reg = "${resample_dir[0]}/Atlas/Native/hemi-${hemi_short}_desc-sphereMSMSulc.surf.gii"

    if ( params.process_stage == 'all' || params.process_stage == 'surface' )
      """
      #!/bin/bash
      bash ${params.surface_script_dir}/scripts/MSMReg.sh \
          --subject_id "${subject_id}" \
          --atlas_space_folder "${resample_dir[0]}/Atlas" \
          --native_folder Native \
          --freesurfer_folder "${surf_dir}" \
          --surface_atlas_dir "${params.surface_atlas_dir}" \
          --high_res_mesh ${params.high_res_mesh} \
          --reg_name ${params.reg_name} \
          --caret7_dir "${params.caret7dir}" \
          --msm_bindir "${params.msmbindir}" \
          --msm_configdir "${params.msmconfigdir}" \
          --hemi ${hemi}
      """
    else
        """echo skip surface"""
}

process pialsurf {
    tag { "${ses_key}::${hemi}" }
    cpus 2

    input:
    tuple val(ses_key), val(hemi), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(white_prehires),
          val(t1w_hires_white), val(t1w_hires_pial_one), val(t1w_hires_pial_two), val(t2w_hires_pial)
    
    output:
    tuple val(ses_key), val(hemi),
          val(pial_hires)
    
    script:
    subject_id = "workspace"
    surf_dir = "${surface_dir[0]}/${subject_id}/surf"
    pial_hires = "${surf_dir}/${hemi}.pial"

    if ( params.process_stage == 'all' || params.process_stage == 'surface' || params.process_stage == 'pial' )
      """
      #!/bin/bash
      bash ${params.surface_script_dir}/scripts/HiresPial.sh \
          --subject_dir ${surface_dir[0]} \
          --subject_id ${subject_id} \
          --hemi ${hemi} \
          --enable_t2 ${params.t2_refine_pial}
      """
    else
        """echo skip surface"""
}

workflow surface {
    take:
    qc_dir
    ses_data_in

    main:
    hemi = channel.of('lh', 'rh')
    ses_info = ses_data_in.combine(hemi).map {
      ses_key, subj_list, ses_list, modality_list, prepare_dir, enhance_dir, surface_dir, resample_dir, t1w_acpc_iso, t2w_acpc_iso, acpc_mat, t1w_acpc_complete_aseg, t1w_white_skeleton, hemi_id ->
      tuple(ses_key, hemi_id, subj_list, ses_list, modality_list, prepare_dir, enhance_dir, surface_dir, resample_dir)
    }

    ses_pretess_out = pretess(ses_data_in) // filled, aseg
    
    ses_tesselate_in = ses_info.combine(ses_pretess_out, by: 0)
    ses_tesselate_out = tessellate(ses_tesselate_in) // white_prehires
    
    ses_prehires_in = ses_info.combine(ses_tesselate_out, by: [0, 1]).groupTuple(by: 0)
    ses_prehires_out = prehires(ses_prehires_in) // t1w_hires_white, t1w_hires_pial_one, t1w_hires_pial_two, t2w_hires_pial

    ses_whitesurf_in = ses_info.combine(ses_tesselate_out, by: [0, 1]).combine(ses_prehires_out, by: 0)
    ses_whitesurf_out = whitesurf(ses_whitesurf_in) // white_hires

    ses_pialsurf_in = ses_info.combine(ses_tesselate_out, by: [0, 1]).combine(ses_prehires_out, by: 0)
    ses_pialsurf_out = pialsurf(ses_pialsurf_in) // pial_hires

    ses_prereg_in = ses_info.combine(ses_whitesurf_out, by: [0, 1]).groupTuple(by: 0)
    ses_prereg_out = prereg(ses_prereg_in) // lh_medialwall, rh_medialwall

    ses_surfreg_in = ses_info.combine(ses_prereg_out, by: 0)
    ses_surfreg_out = surfreg(ses_surfreg_in) // fs_sphere_reg

    // msmreg needs pial thickness, so must wait for pialsurf to finish
    ses_msmreg_in = ses_info.combine(ses_surfreg_out, by: [0, 1])
        .combine(ses_pialsurf_out, by: [0, 1])
        .map { sk, h, sl, sel, ml, pd, ed, sd, rd, fsr, ph ->
            tuple(sk, h, sl, sel, ml, pd, ed, sd, rd, fsr)
        }
    ses_msmreg_out = msmreg(ses_msmreg_in) // msm_sphere_reg

    emit:
    ses_info
    ses_msmreg_out
    ses_whitesurf_out
    ses_pialsurf_out
}

process acpc_ribbon {
    tag { "${ses_key}" }
    cpus 1

    input:
    tuple val(ses_key), val(hemi_list), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(white_hires), val(pial_hires)
    
    output:
    tuple val(ses_key), 
          val(acpc_ribbon)
    
    script:
    def (subj, ses) = ses_key.split('::')
    subject_id = "workspace"
    t1w_acpc_0mm = "${enhance_dir[0][0]}/T1w/${subj}_${ses}_space-acpc_desc-head_T1w.nii.gz"
    t1w_acpc_1mm = "${surface_dir[0][0]}/${subject_id}/mri/T1w_hires.nii.gz"
    acpc_ribbon = "${resample_dir[0][0]}/ACPC/Volume/${subj}_${ses}_desc-ribbon_dseg.nii.gz"

    if ( params.process_stage == 'all' || params.process_stage == 'resample' )
      """
      #!/bin/bash
      set -e
      set -x
      bash ${params.resample_script_dir}/scripts/AcpcRibbon.sh \
          --preprocess_dir ${enhance_dir[0][0]} \
          --freesurfer_dir ${surface_dir[0][0]}/${subject_id} \
          --resample_dir ${resample_dir[0][0]} \
          --t1w_image_0mm ${t1w_acpc_0mm} \
          --t1w_image_1mm ${t1w_acpc_1mm} \
          --caret7_dir ${params.caret7dir} \
          --python_inter "${params.python_inter}" \
          --utils_path "${params.utils_path}" \
          --prefix "${subj}_${ses}"
      """
    else
        """echo skip resample"""
}

// -------------------------
//  Check surface
// -------------------------
process qc_surface {
    tag { "${ses_key}" }
    cpus 1

    input:
      tuple val(ses_key), val(hemi_list), val(subj_list), val(ses_list), val(modality_list),
            val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
            val(acpc_ribbon)
      val qc_dir

    output:
      tuple val(ses_key), val(qc_done)

    script:
    qc_done = 1
    def (subject_id, session_id) = ses_key.split('::')
    acpc_brain = "${resample_dir[0][0]}/ACPC/Volume/${subject_id}_${session_id}_space-acpc_desc-brain_T1w.nii.gz"
    session_qc_dir = "${qc_dir}/${subject_id}/${session_id}"
    qc_output = "${session_qc_dir}/qc_surface.png"
    if ( params.process_stage == 'all' || params.process_stage == 'resample' )
      """
      ${params.python_inter} ${params.utils_path}/qc_surface.py \
        --input ${acpc_brain} \
        --ribbon ${acpc_ribbon[0]} \
        --output ${qc_output} \
        --single_contour
      """
    else
        """echo skip resample"""
}

process orig_ribbon {
    tag { "${ses_key}" }
    cpus 1

    input:
    tuple val(ses_key), val(hemi_list), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(acpc_ribbon)
    
    output:
    tuple val(ses_key), 
          val(orig_ribbon)
    
    script:
    def (subj, ses) = ses_key.split('::')
    subject_id = "workspace"
    orig_ribbon = "${resample_dir[0][0]}/Original/Volume/${subj}_${ses}_desc-ribbon_dseg.nii.gz"

    if ( params.process_stage == 'all' || params.process_stage == 'resample' )
      """
      #!/bin/bash
      set -e
      set -x
      bash ${params.resample_script_dir}/scripts/OrigRibbon.sh \
          --preprocess_dir ${enhance_dir[0][0]} \
          --freesurfer_dir ${surface_dir[0][0]}/${subject_id} \
          --resample_dir ${resample_dir[0][0]} \
          --original_vol ${enhance_dir[0][0]}/T1w/${subj}_${ses}_desc-conform_T1w.nii.gz \
          --acpc_vol ${resample_dir[0][0]}/ACPC/Volume/${subj}_${ses}_space-acpc_desc-brain_T1w.nii.gz \
          --caret7_dir ${params.caret7dir} \
          --python_inter "${params.python_inter}" \
          --utils_path "${params.utils_path}" \
          --prefix "${subj}_${ses}"
      """
    else
        """echo skip resample"""
}

process atlas_resample {
    tag { "${ses_key}::${hemi}" }
    cpus 1

    input:
    tuple val(ses_key), val(hemi), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(orig_ribbon), val(msm_sphere_reg)
    
    output:
    tuple val(ses_key), val(hemi),
          val(atlas_surface)
    
    script:
    def (subj, ses) = ses_key.split('::')
    subject_id = "workspace"
    hemi_short = hemi == 'lh' ? 'L' : 'R'
    atlas_surface = "${resample_dir[0]}/Atlas/Native/${subj}_${ses}_hemi-${hemi_short}_desc-pial.surf.gii"

    if ( params.process_stage == 'all' || params.process_stage == 'resample' )
      """
      #!/bin/bash
      set -e
      set -x
      bash ${params.resample_script_dir}/scripts/TempResample.sh \
          --preprocess_dir ${enhance_dir[0]} \
          --freesurfer_dir ${surface_dir[0]}/${subject_id} \
          --resample_dir ${resample_dir[0]} \
          --surface_atlas_dir ${params.surface_atlas_dir} \
          --hemi ${hemi} \
          --low_res_meshes ${params.low_res_mesh} \
          --regname ${params.reg_name} \
          --high_res_mesh ${params.high_res_mesh} \
          --python_inter "${params.python_inter}" \
          --utils_path "${params.utils_path}" \
          --caret7_dir ${params.caret7dir} \
          --prefix "${subj}_${ses}"
      """
    else
        """echo skip resample"""
}

process acpc_resample {
    tag { "${ses_key}::${hemi}" }
    cpus 1

    input:
    tuple val(ses_key), val(hemi), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(atlas_surface)
    
    output:
    tuple val(ses_key), val(hemi),
          val(acpc_surface)
    
    script:
    def (subj, ses) = ses_key.split('::')
    subject_id = "workspace"
    hemi_short = hemi == 'lh' ? 'L' : 'R'
    acpc_surface = "${resample_dir[0]}/ACPC/Native/${subj}_${ses}_hemi-${hemi_short}_desc-pial.surf.gii"

    if ( params.process_stage == 'all' || params.process_stage == 'resample' )
      """
      #!/bin/bash
      set -e
      set -x
      bash ${params.resample_script_dir}/scripts/AcpcResample.sh \
          --freesurfer_dir ${surface_dir[0]}/${subject_id} \
          --resample_dir ${resample_dir[0]} \
          --hemi ${hemi} \
          --low_res_meshes ${params.low_res_mesh} \
          --python_inter "${params.python_inter}" \
          --utils_path "${params.utils_path}" \
          --caret7_dir ${params.caret7dir} \
          --prefix "${subj}_${ses}"
      """
    else
        """echo skip resample"""
}

process orig_resample {
    tag { "${ses_key}::${hemi}" }
    cpus 1

    input:
    tuple val(ses_key), val(hemi), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(acpc_surface)
    
    output:
    tuple val(ses_key), val(hemi),
          val(orig_surface)
    
    script:
    def (subj, ses) = ses_key.split('::')
    subject_id = "workspace"
    hemi_short = hemi == 'lh' ? 'L' : 'R'
    orig_surface = "${resample_dir[0]}/Original/Native/${subj}_${ses}_hemi-${hemi_short}_desc-pial.surf.gii"

    if ( params.process_stage == 'all' || params.process_stage == 'resample' )
      """
      #!/bin/bash
      set -e
      set -x
      bash ${params.resample_script_dir}/scripts/OrigResample.sh \
          --freesurfer_dir ${surface_dir[0]}/${subject_id} \
          --resample_dir ${resample_dir[0]} \
          --hemi ${hemi} \
          --low_res_meshes ${params.low_res_mesh} \
          --python_inter "${params.python_inter}" \
          --utils_path "${params.utils_path}" \
          --caret7_dir ${params.caret7dir} \
          --prefix "${subj}_${ses}"
      """
    else
        """echo skip resample"""
}

process annot {
    tag { "${ses_key}" }
    accelerator 1
    maxForks 1 * ((params.per_gpu == null) ? 1 : (params.per_gpu as int))

    input:
    tuple val(ses_key), val(hemi_list), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(acpc_surface), val(orig_surface)
    val qc_dir
    
    output:
    tuple val(ses_key), 
          val(annotation)
    
    script:
    def (subject_id, session_id) = ses_key.split('::')
    annotation = "${resample_dir[0][0]}/${subject_id}_${session_id}_hemi-L_desc-aparc.label.gii"

    if ( params.process_stage == 'all' || params.process_stage == 'resample' )
      """
      #!/bin/bash
      set -e
      set -x
      bash ${params.resample_script_dir}/scripts/Annot.sh \
          --preprocess_dir "${enhance_dir[0][0]}" \
          --resample_dir "${resample_dir[0][0]}" \
          --python_inter "${params.python_inter}" \
          --surf_reg_dir "${params.surf_reg_dir}" \
          --template_dir "${params.hcppipedir_templates}/MEBRAIN" \
          --utils_path "${params.utils_path}" \
          --prefix "${subject_id}_${session_id}"

      bash ${params.resample_script_dir}/scripts/NormStatsPredict.sh \
          --subjects_dir "${params.out_dir}" \
          --subject "${subject_id}" \
          --session "${session_id}" \
          --out_dir "${params.out_dir}/${subject_id}/${session_id}/Stats" \
          --meta_csv "${qc_dir}/meta.csv" \
          --python_inter "${params.python_inter}" \
          --utils_path "${params.utils_path}" \
          --norm_model_dir "${params.norm_model_path}" \
          --atlases_dir "${params.atlases_dir}" \
          --atlases "${params.cortical_atlases}" \
      """
    else
        """echo skip resample"""
}



workflow resample {
    take:
    qc_dir
    ses_info
    ses_msmreg_out
    ses_whitesurf_out
    ses_pialsurf_out

    main:
    ses_acpc_ribbon_in = ses_info.combine(ses_whitesurf_out, by: [0, 1]).combine(ses_pialsurf_out, by: [0, 1]).groupTuple(by: 0)
    ses_acpc_ribbon_out = acpc_ribbon(ses_acpc_ribbon_in) // acpc_ribbon

    ses_qc_surface_in = ses_info.combine(ses_acpc_ribbon_out, by: 0).groupTuple(by: 0)
    ses_qc_surface_out = qc_surface(ses_qc_surface_in, qc_dir)

    ses_orig_ribbon_in = ses_info.combine(ses_acpc_ribbon_out, by: 0).groupTuple(by: 0)
    ses_orig_ribbon_out = orig_ribbon(ses_orig_ribbon_in)
    
    ses_atlas_resample_in = ses_info.combine(ses_orig_ribbon_out, by: 0).combine(ses_msmreg_out, by: [0, 1])
    ses_atlas_resample_out = atlas_resample(ses_atlas_resample_in)
    
    ses_acpc_resample_in = ses_info.combine(ses_atlas_resample_out, by: [0, 1])
    ses_acpc_resample_out = acpc_resample(ses_acpc_resample_in)
    
    ses_orig_resample_in = ses_info.combine(ses_acpc_resample_out, by: [0, 1])
    ses_orig_resample_out = orig_resample(ses_orig_resample_in)

    ses_annot_in = ses_info.combine(ses_acpc_resample_out, by: [0, 1]).combine(ses_orig_resample_out, by: [0, 1]).groupTuple(by: 0)
    ses_annot_out = annot(ses_annot_in, qc_dir)
    
    emit:
    ses_annot_out
    ses_qc_surface_out
}

// ========================================================================
// BOLD fMRI PREPROCESSING
// ========================================================================

process bold_get_func {
    tag "BOLD discovery"
    cpus 1
    memory '500 MB'

    input:
    val bids_dir
    val participant_label
    val session_id
    val bold_task_type
    val bold_only

    output:
    path "BOLD/sub-*"  // per-subject job directories

    script:
    """
    bash ${params.hcppipedir}/nextflow/BOLD/scripts/bold_get_func.sh \
        --bids_dir ${bids_dir} \
        --out_dir . \
        --participant_label '${participant_label}' \
        --session_id '${session_id}' \
        --bold_task_type '${bold_task_type}' \
        --bold_only ${bold_only}
    """
}

process bold_anat_prepare {
    tag "${ses_key}"
    cpus 1

    input:
    tuple val(ses_key), val(subj_list), val(ses_list), val(modality_list),
          val(prepare_dir), val(enhance_dir), val(surface_dir), val(resample_dir),
          val(t1w_acpc_iso), val(t2w_acpc_iso), val(acpc_mat),
          val(t1w_acpc_complete_aseg), val(t1w_white_skeleton)
    val bold_path

    output:
    tuple val(ses_key),
          val(t1w_ref), val(mask_ref), val(wm_prob), val(gm_prob), val(csf_prob), val(wm_dseg), val(fsnative2t1w_xfm)

    script:
    def subj = subj_list[0]
    def ses = ses_list[0]
    def ses_prefix = ses ? "_${ses}" : ""
    def file_prefix = "${subj}${ses_prefix}"
    def bold_session_path = ses ? "${bold_path}/${subj}/${ses}/BOLD" : "${bold_path}/${subj}/BOLD"
    t1w_conform = "${enhance_dir[0]}/T1w/${file_prefix}_desc-conform_T1w.nii.gz"
    brainmask   = "${enhance_dir[0]}/T1w/${file_prefix}_desc-conform_mask.nii.gz"
    nbest_seg   = "${enhance_dir[0]}/T1w/${file_prefix}_desc-nbest_dseg.nii.gz"
    t2w_conform = "${enhance_dir[0]}/T1w/${file_prefix}_desc-conform_T2w.nii.gz"

    t1w_ref = "${bold_session_path}/anat/${file_prefix}_desc-preproc_T1w.nii.gz"
    mask_ref = "${bold_session_path}/anat/${file_prefix}_desc-brain_mask.nii.gz"
    wm_prob = "${bold_session_path}/anat/${file_prefix}_label-WM_probseg.nii.gz"
    gm_prob = "${bold_session_path}/anat/${file_prefix}_label-GM_probseg.nii.gz"
    csf_prob = "${bold_session_path}/anat/${file_prefix}_label-CSF_probseg.nii.gz"
    wm_dseg = "${bold_session_path}/anat/${file_prefix}_label-WM_dseg.nii.gz"
    fsnative2t1w_xfm = "${bold_session_path}/anat/${file_prefix}_from-fsnative_to-T1w_mode-image_xfm.txt"

    if ( params.process_stage == 'all' || params.process_stage == 'bold' )
      """
      mkdir -p "${bold_session_path}/anat"

      bash ${params.hcppipedir}/nextflow/BOLD/scripts/bold_anat_prepare.sh \
          --bold_path ${bold_session_path} \
          --subj ${subj} \
          --ses ${ses} \
          --t1w_conform ${t1w_conform} \
          --brainmask ${brainmask} \
          --nbest_seg ${nbest_seg} \
          --t2w_conform ${t2w_conform}
      """
    else
        """echo skip bold anat_prepare"""
}

process bold_preprocess {
    tag "${bold_id}"
    cpus 4
    memory '4 GB'

    input:
    tuple val(ses_key), val(bold_id), path(bold_job_file),
          val(t1w_ref), val(mask_ref), val(wm_dseg), val(wm_prob), val(gm_prob), val(csf_prob),
          val(subjects_dir)
    val fs_license_file
    val bold_preprocess_dir
    val qc_dir
    val skip_frame
    val sdc
    val reg_method
    val bids_dir

    output:
    tuple val(ses_key), val(bold_id), path("func/${bold_id}_space-T1w_desc-preproc_bold.nii.gz"), path(bold_job_file)

    script:
    def (proc_subj, proc_ses) = ses_key.split('::')
    def bold_session_dir = proc_ses ? "${bold_preprocess_dir}/${proc_subj}/${proc_ses}/BOLD" : "${bold_preprocess_dir}/${proc_subj}/BOLD"
    """
    # Read job file
    bold_orig=\$(sed -n '2p' ${bold_job_file})
    subj=\$(sed -n '1p' ${bold_job_file})
    ses=\$(echo "${bold_id}" | grep -oP 'ses-[^_]+' || echo "ses-001")

    bash ${params.hcppipedir}/nextflow/BOLD/scripts/bold_preprocess.sh \
        --bold_file "\${bold_orig}" \
        --bold_preprocess_dir ${bold_session_dir} \
        --subj "\${subj}" \
        --ses "\${ses}" \
        --bids_dir ${bids_dir} \
        --t1w_ref ${t1w_ref} \
        --t1w_mask ${mask_ref} \
        --t1w_dseg ${wm_dseg} \
        --subjects_dir ${subjects_dir} \
        --fs_license ${fs_license_file} \
        --skip_frame ${skip_frame} \
        --sdc ${sdc} \
        --reg_method ${reg_method} \\
        --viz true

    # Copy outputs from session dir to work dir for Nextflow
    mkdir -p func
    cp -r ${bold_session_dir}/func/${bold_id}_space-T1w_desc-preproc_bold.nii.gz func/ 2>/dev/null || true
    """
}

process bold_fieldmap_estimate {
    tag "${subj}${ses}"
    cpus 2
    memory '2 GB'

    input:
    val bids_dir
    val bold_preprocess_dir
    tuple val(ses_key), val(subj), val(ses)
    val sdc

    output:
    tuple val(ses_key), path("fmap_done.txt")

    script:
    def bold_session_dir = ses ? "${bold_preprocess_dir}/${subj}/${ses}/BOLD" : "${bold_preprocess_dir}/${subj}/BOLD"
    if (sdc.toString().toUpperCase() == 'TRUE') {
        """
        ${params.python_inter} ${params.hcppipedir}/nextflow/BOLD/scripts/bold_fieldmap_estimate.py \\
            --bids_dir ${bids_dir} \\
            --bold_preprocess_dir ${bold_session_dir} \\
            --subject_id ${subj} \\
            --session_id ${ses} \\
            --omp_nthreads 2

        echo "done" > fmap_done.txt
        """
    } else {
        """
        echo "SDC disabled; skipping fieldmap estimation"
        echo "skipped" > fmap_done.txt
        """
    }
}

process bold_confounds {
    tag "${bold_id}"
    cpus 2
    memory '2 GB'

    input:
    tuple val(ses_key), val(bold_id), val(t1w_space_bold), val(bold_job_file),
          val(t1w_ref), val(mask_ref), val(wm_dseg), val(wm_prob), val(gm_prob), val(csf_prob)
    val bold_preprocess_dir
    val work_dir
    val skip_frame
    val bandpass

    output:
    tuple val(ses_key), val(bold_id), path("func/${bold_id}_desc-confounds_timeseries.tsv")

    script:
    def (conf_subj, conf_ses) = ses_key.split('::')
    def bold_session_dir = conf_ses ? "${bold_preprocess_dir}/${conf_subj}/${conf_ses}/BOLD" : "${bold_preprocess_dir}/${conf_subj}/BOLD"
    """
    bash ${params.hcppipedir}/nextflow/BOLD/scripts/bold_confounds.sh \
        --bold_preprocess_dir ${bold_session_dir} \
        --work_dir ${work_dir}/bold_confounds \
        --subj ${conf_subj} \
        --ses ${conf_ses} \
        --bold_id ${bold_id} \
        --t1w_ref ${t1w_ref} \
        --t1w_mask ${mask_ref} \
        --wm_prob ${wm_prob} \
        --csf_prob ${csf_prob} \
        --nbest_seg ${params.hcppipedir}/none \
        --skip_frame ${skip_frame} \
        --bandpass ${bandpass} \
        --python_inter ${params.python_inter} \
        --utils_path ${params.utils_path}

    mkdir -p func
    cp -r ${bold_session_dir}/func/${bold_id}_desc-confounds_timeseries.tsv func/ 2>/dev/null || true
    """
}

process bold_normalize {
    tag "${bold_id}"
    cpus 1
    memory '2 GB'

    input:
    tuple val(ses_key), val(bold_id), path(bold_job_file),
          val(t1w_ref), val(mask_ref), val(wm_dseg), val(wm_prob), val(gm_prob), val(csf_prob),
          val(warp_file)
    val template_space
    val bold_preprocess_dir

    output:
    tuple val(ses_key), val(bold_id), path("func/${bold_id}_space-${template_space}_desc-preproc_bold.nii.gz")

    script:
    def (norm_subj, norm_ses) = ses_key.split('::')
    def bold_session_dir = norm_ses ? "${bold_preprocess_dir}/${norm_subj}/${norm_ses}/BOLD" : "${bold_preprocess_dir}/${norm_subj}/BOLD"
    """
    bash ${params.hcppipedir}/nextflow/BOLD/scripts/bold_normalize.sh \
        --bold_preprocess_dir ${bold_session_dir} \
        --subj ${norm_subj} \
        --ses ${norm_ses} \
        --bold_id ${bold_id} \
        --warp_file ${warp_file} \
        --t1w_ref ${t1w_ref} \
        --template_space ${template_space}

    mkdir -p func
    cp -r ${bold_session_dir}/func/${bold_id}_space-${template_space}_desc-preproc_bold.nii.gz func/ 2>/dev/null || true
    """
}

process bold_vol2surf {
    tag "${bold_id}::${hemi}"
    cpus 1
    memory '2 GB'

    input:
    tuple val(ses_key), val(bold_id), path(bold_job_file),
          val(t1w_ref), val(mask_ref), val(wm_dseg), val(wm_prob), val(gm_prob), val(csf_prob),
          val(white_L), val(white_R), val(pial_L), val(pial_R), val(ribbon)
    each hemi
    val bold_preprocess_dir

    output:
    tuple val(ses_key), val(bold_id), val(hemi), path("surf/${bold_id}_hemi-${hemi}_space-fsnative_bold.func.gii")

    script:
    def white_surf = hemi == 'L' ? white_L : white_R
    def pial_surf = hemi == 'L' ? pial_L : pial_R
    def (surf_subj, surf_ses) = ses_key.split('::')
    def bold_session_dir = surf_ses ? "${bold_preprocess_dir}/${surf_subj}/${surf_ses}/BOLD" : "${bold_preprocess_dir}/${surf_subj}/BOLD"
    """
    bash ${params.hcppipedir}/nextflow/BOLD/scripts/bold_vol2surf.sh \
        --bold_preprocess_dir ${bold_session_dir} \
        --subj ${surf_subj} \
        --ses ${surf_ses} \
        --bold_id ${bold_id} \
        --hemi ${hemi} \
        --white_surf ${white_surf} \
        --pial_surf ${pial_surf} \
        --ribbon ${ribbon} \
        --caret7_dir ${params.caret7dir}

    mkdir -p surf
    cp -r ${bold_session_dir}/surf/${bold_id}_hemi-${hemi}_space-fsnative_bold.func.gii surf/ 2>/dev/null || true
    """
}

process qc_bold_motion {
    tag "${bold_id} - motion"
    cpus 1
    memory '2 GB'

    input:
    tuple val(ses_key), val(bold_id), val(t1w_ref),
          val(bold_preprocess_dir)
    val qc_dir

    output:
    tuple val(ses_key), path("qc_bold_motion_${bold_id}.png")

    script:
    def (qc_subj, qc_ses) = ses_key.split('::')
    def bold_session_dir = qc_ses ? "${bold_preprocess_dir}/${qc_subj}/${qc_ses}/BOLD" : "${bold_preprocess_dir}/${qc_subj}/BOLD"
    def session_qc_dir = "${qc_dir}/${qc_subj}/${qc_ses}"
    def motion_params = "${bold_session_dir}/func/${bold_id}_desc-mc_motion_params.txt"
    def rms_rel = "${bold_session_dir}/func/${bold_id}_desc-mc_rel.rms"
    def rms_abs = "${bold_session_dir}/func/${bold_id}_desc-mc_abs.rms"
    """
    mkdir -p "${session_qc_dir}"
    ${params.python_inter} ${params.utils_path}/qc_bold_motion.py \
        --motion-params "${motion_params}" \
        --rms-rel "${rms_rel}" \
        --rms-abs "${rms_abs}" \
        --output "${session_qc_dir}/qc_bold_motion_${bold_id}.png"

    cp "${session_qc_dir}/qc_bold_motion_${bold_id}.png" . 2>/dev/null || true
    """
}

process qc_bold_registration {
    tag "${bold_id} - reg"
    cpus 1
    memory '2 GB'

    input:
    tuple val(ses_key), val(bold_id), val(t1w_ref),
          val(bold_preprocess_dir)
    val qc_dir

    output:
    tuple val(ses_key), path("qc_bold_registration_${bold_id}.png")

    script:
    def (qc_subj, qc_ses) = ses_key.split('::')
    def bold_session_dir = qc_ses ? "${bold_preprocess_dir}/${qc_subj}/${qc_ses}/BOLD" : "${bold_preprocess_dir}/${qc_subj}/BOLD"
    def session_qc_dir = "${qc_dir}/${qc_subj}/${qc_ses}"
    def bold_t1w = "${bold_session_dir}/func/${bold_id}_space-T1w_desc-preproc_bold.nii.gz"
    """
    mkdir -p "${session_qc_dir}"
    ${params.python_inter} ${params.utils_path}/qc_bold_registration.py \
        --bold-t1w "${bold_t1w}" \
        --t1w-ref ${t1w_ref} \
        --output "${session_qc_dir}/qc_bold_registration_${bold_id}.png"

    cp "${session_qc_dir}/qc_bold_registration_${bold_id}.png" . 2>/dev/null || true
    """
}

process qc_bold_tsnr {
    tag "${bold_id} - tsnr"
    cpus 1
    memory '2 GB'

    input:
    tuple val(ses_key), val(bold_id), val(t1w_ref),
          val(bold_preprocess_dir)
    val qc_dir

    output:
    tuple val(ses_key), path("qc_bold_tsnr_${bold_id}.png")

    script:
    def (qc_subj, qc_ses) = ses_key.split('::')
    def bold_session_dir = qc_ses ? "${bold_preprocess_dir}/${qc_subj}/${qc_ses}/BOLD" : "${bold_preprocess_dir}/${qc_subj}/BOLD"
    def session_qc_dir = "${qc_dir}/${qc_subj}/${qc_ses}"
    def bold_preproc = "${bold_session_dir}/func/${bold_id}_desc-preproc_bold.nii.gz"
    def bold_mask = "${bold_session_dir}/func/${bold_id}_desc-brain_mask.nii.gz"
    """
    mkdir -p "${session_qc_dir}"
    ${params.python_inter} ${params.utils_path}/qc_bold_tsnr.py \
        --bold-preproc "${bold_preproc}" \
        --bold-mask "${bold_mask}" \
        --output "${session_qc_dir}/qc_bold_tsnr_${bold_id}.png"

    cp "${session_qc_dir}/qc_bold_tsnr_${bold_id}.png" . 2>/dev/null || true
    """
}

process qc_bold_carpet {
    tag "${bold_id} - carpet"
    cpus 1
    memory '2 GB'

    input:
    tuple val(ses_key), val(bold_id), val(t1w_ref),
          val(bold_preprocess_dir)
    val qc_dir

    output:
    tuple val(ses_key), path("qc_bold_carpet_${bold_id}.png")

    script:
    def (qc_subj, qc_ses) = ses_key.split('::')
    def bold_session_dir = qc_ses ? "${bold_preprocess_dir}/${qc_subj}/${qc_ses}/BOLD" : "${bold_preprocess_dir}/${qc_subj}/BOLD"
    def session_qc_dir = "${qc_dir}/${qc_subj}/${qc_ses}"
    def bold_preproc = "${bold_session_dir}/func/${bold_id}_desc-preproc_bold.nii.gz"
    def bold_mask = "${bold_session_dir}/func/${bold_id}_desc-brain_mask.nii.gz"
    def motion_params = "${bold_session_dir}/func/${bold_id}_desc-mc_motion_params.txt"
    """
    mkdir -p "${session_qc_dir}"
    ${params.python_inter} ${params.utils_path}/qc_bold_carpet.py \
        --bold-preproc "${bold_preproc}" \
        --bold-mask "${bold_mask}" \
        --motion-params "${motion_params}" \
        --output "${session_qc_dir}/qc_bold_carpet_${bold_id}.png"

    cp "${session_qc_dir}/qc_bold_carpet_${bold_id}.png" . 2>/dev/null || true
    """
}

process qc_bold_normalize {
    tag "${bold_id} - norm"
    cpus 1
    memory '2 GB'

    input:
    tuple val(ses_key), val(bold_id), path(bold_template),
          val(t1w_ref), val(bold_preprocess_dir)
    val qc_dir

    output:
    tuple val(ses_key), path("qc_bold_normalize_${bold_id}.png")

    script:
    def (qc_subj, qc_ses) = ses_key.split('::')
    def bold_session_dir = qc_ses ? "${bold_preprocess_dir}/${qc_subj}/${qc_ses}/BOLD" : "${bold_preprocess_dir}/${qc_subj}/BOLD"
    def session_qc_dir = "${qc_dir}/${qc_subj}/${qc_ses}"
    """
    mkdir -p "${session_qc_dir}"
    ${params.python_inter} ${params.utils_path}/qc_bold_normalize.py \
        --bold-template "${bold_template}" \
        --template-ref ${t1w_ref} \
        --output "${session_qc_dir}/qc_bold_normalize_${bold_id}.png"

    cp "${session_qc_dir}/qc_bold_normalize_${bold_id}.png" . 2>/dev/null || true
    """
}

process qc_bold_surf {
    tag "${bold_id} - surf ${hemi}"
    cpus 1
    memory '2 GB'

    input:
    tuple val(ses_key), val(bold_id), val(hemi), path(surf_gii),
          val(ribbon), val(bold_preprocess_dir)
    val qc_dir

    output:
    tuple val(ses_key), path("qc_bold_surf_${bold_id}_hemi-${hemi}.png")

    script:
    def (qc_subj, qc_ses) = ses_key.split('::')
    def session_qc_dir = "${qc_dir}/${qc_subj}/${qc_ses}"
    """
    mkdir -p "${session_qc_dir}"
    ${params.python_inter} ${params.utils_path}/qc_bold_surf.py \
        --surf-gii "${surf_gii}" \
        --ribbon ${ribbon} \
        --output "${session_qc_dir}/qc_bold_surf_${bold_id}_hemi-${hemi}.png"

    cp "${session_qc_dir}/qc_bold_surf_${bold_id}_hemi-${hemi}.png" . 2>/dev/null || true
    """
}

// ========================================================================
// BOLD WORKFLOW
// ========================================================================
workflow bold_wf {

    take:
    qc_dir
    ses_data_in        // struct outputs from surface/enhance stage
    resample_complete  // dependency signal: ensures resample finished before vol2surf

    main:
    bids_dir = params.bids_dir
    out_dir = params.out_dir
    participant_label = params.participant_label
    session_id = params.session_id
    bold_task_type = params.bold_task_type
    bold_skip_frame = params.bold_skip_frame
    bold_bandpass = params.bold_bandpass
    bold_sdc = params.bold_sdc
    bold_reg_method = params.bold_reg_method
    bold_only = params.bold_only
    bold_cifti = params.bold_cifti
    do_bold_confounds = params.bold_confounds
    template_space = params.bold_volume_space
    subjects_dir = "${out_dir}/${participant_label}/Surface"
    fs_license_file = params.fs_license_file

    // Step 1: Discover BOLD files
    bold_files = bold_get_func(bids_dir, participant_label, session_id, bold_task_type, bold_only)

    // Build session+BOLD channel
    work_dir = "${params.out_dir}/WorkDir/bold"
    bold_preprocess_path = "${params.out_dir}"
    hemi_ch = Channel.of('L', 'R')

    // Step 2: Anatomical preparation (bridge)
    bold_anat_out = bold_anat_prepare(ses_data_in, bold_preprocess_path)

    // Combine session info with BOLD files
    // bold_files are per-subject dirs with job files inside
    // Flatten directory outputs into individual job files
    // bold_files emits subject directories; flatMap lists files containing '_bold'
    bold_job_files = bold_files
        .flatMap { dir ->
            def d = dir.toFile()
            if (d.isDirectory()) {
                d.listFiles()?.findAll { it.name.contains('_bold') }?.collect { file(it) } ?: []
            } else {
                []
            }
        }

    // For each BOLD run, pair with session's structural data
    // Extract subject AND session from job file path/name
    // Use single join key: "subj::ses" (string) to avoid Nextflow's one-to-one
    // composite-key join behavior
    bold_with_ses = bold_job_files
        .map { job_file ->
            def path_str = job_file.toString()
            def subj = path_str.contains('/sub-') ? (path_str =~ /\/sub-[^\/]+/)[0].replace('/', '') : 'sub-unknown'
            // Extract session from filename: e.g., sub-032144_ses-004_task-resting_acq-RL_run-1_bold → ses-004
            def ses_match = (job_file.name =~ /ses-[^_]+/)
            def ses = ses_match ? ses_match[0] : ''
            def join_key = ses ? "${subj}::${ses}" : subj
            // bold_preprocess.sh internally strips _bold.nii.gz from the file path,
            // so bold_id must match (strip _bold suffix from the job filename)
            def bold_id = job_file.name.replaceFirst(/_bold$/, '')
            tuple(join_key, subj, ses, bold_id, job_file)
        }

    // Join BOLD runs with structural data via subject::session string key
    // Emit: (join_key, subj, ses, ses_key, enhance_dir, resample_dir)
    ses_data_flat = ses_data_in.map { ses_key, subj_list, ses_list, modality_list, prep, enh, surf, resamp,
                                        t1w_acpc_iso, t2w_acpc_iso, acpc_mat, complete_aseg, wm_skeleton ->
        def subj = subj_list[0]
        def ses = ses_list[0]
        def join_key = ses ? "${subj}::${ses}" : subj
        tuple(join_key, subj, ses, ses_key, enh[0], resamp[0])
    }

    // Extract anatomical bridge outputs
    bold_anat_flat = bold_anat_out.map { ses_key, t1w_ref, mask_ref, wm_prob, gm_prob, csf_prob, wm_dseg, xfm ->
        tuple(ses_key, t1w_ref, mask_ref, wm_prob, gm_prob, csf_prob, wm_dseg, xfm)
    }

    // Join: BOLD runs + session data (on join_key) + anat data (on ses_key)
    // Use combine() + filter() instead of combine(by:) because combine(by:) has
    // timing issues when right-hand channel emits after left-hand channel.
    //
    // bold_with_ses: (join_key, subj, ses, bold_id, job_file)            -- 5 elts
    // ses_data_flat: (join_key, subj, ses, ses_key, enhance_dir, resample_dir) -- 6 elts
    // Cartesian product then filter by matching join_key
    bold_struct_join1 = bold_with_ses.combine(ses_data_flat)
        .filter { jk1, sb1, se1, bid, jf, jk2, sb2, se2, sk, ed, rd -> jk1 == jk2 }
        .map { jk1, sb1, se1, bid, jf, jk2, sb2, se2, sk, ed, rd ->
            tuple(jk1, sb1, se1, bid, jf, sk, ed, rd)
        }

    // bold_struct_join1: (join_key, subj, ses, bold_id, job_file, ses_key, enhance_dir, resample_dir) -- 9 elts
    // bold_anat_flat:   (ses_key, t1w_ref, mask_ref, wm_prob, gm_prob, csf_prob, wm_dseg, xfm) -- 8 elts
    // Cartesian product then filter by matching ses_key (index 5 vs index 0)
    bold_struct = bold_struct_join1.combine(bold_anat_flat)
        .filter { jk, sb, se, bid, jf, sk1, ed, rd, sk2, t1w, mask, wm_p, gm_p, csf_p, wm_d, xfm -> sk1 == sk2 }
        .map { jk, sb, se, bid, jf, sk1, ed, rd, sk2, t1w, mask, wm_p, gm_p, csf_p, wm_d, xfm ->
            tuple(jk, sb, se, bid, jf, sk1, ed, rd, t1w, mask, wm_p, gm_p, csf_p, wm_d, xfm)
        }

    // --- Step 2b: Fieldmap estimation (per unique session) ---
    // Extract unique (ses_key, subj, ses) tuples so fieldmap estimation
    // runs once per session, shared across all BOLD runs.
    bold_sessions = bold_struct
        .map { jk, sb, se, bid, jf, sk, ed, rd,
               t1w, mask, wm_p, gm_p, csf_p, wm_d, xfm ->
            tuple(sk, sb, se ?: '')
        }
        .unique()

    bold_fmap_done = bold_fieldmap_estimate(
        bids_dir,
        bold_preprocess_path,
        bold_sessions,
        bold_sdc
    )

    // --- Step 3: Core BOLD preprocessing ---
    // Include per-session subjects_dir (FreeSurfer workspace) in the tuple
    // Join with fieldmap completion to ensure estimation finishes first
    def out_dir_fs = "${params.out_dir}"
    bold_preproc_in = bold_struct.map { jk, sb, se, bid, jf, sk, ed, rd,
                          t1w, mask, wm_p, gm_p, csf_p, wm_d, xfm ->
            def fs_subjects_dir = se ? "${out_dir_fs}/${sb}/${se}/Surface" : "${out_dir_fs}/${sb}/Surface"
            tuple(sk, bid, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p, fs_subjects_dir)
        }
        .combine(bold_fmap_done)
        .filter { sk1, bid, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p, fs_sd,
                  sk2, fmap_signal -> sk1 == sk2 }
        .map { sk1, bid, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p, fs_sd,
               sk2, fmap_signal ->
            tuple(sk1, bid, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p, fs_sd)
        }

    bold_preproc_out = bold_preprocess(
        bold_preproc_in,
        fs_license_file,
        bold_preprocess_path,
        qc_dir,
        bold_skip_frame,
        bold_sdc,
        bold_reg_method,
        bids_dir
    )

    // --- Step 4: Confounds (optional) ---
    if (do_bold_confounds.toString().toUpperCase() == 'TRUE') {
        bold_confounds_out = bold_confounds(
            // bold_preproc_out: (ses_key, bold_id, t1w_bold, job_file)        -- 4 elts
            // bold_anat_flat:   (ses_key, t1w_ref, mask_ref, wm_p, gm_p, csf_p, wm_d, xfm) -- 8 elts
            bold_preproc_out.combine(bold_anat_flat)
                .filter { sk1, bid, t1wb, jf, sk2, t1w, mask, wm_p, gm_p, csf_p, wm_d, xfm -> sk1 == sk2 }
                .map { sk1, bid, t1wb, jf, sk2, t1w, mask, wm_p, gm_p, csf_p, wm_d, xfm ->
                    tuple(sk1, bid, t1wb, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p)
                },
            bold_preprocess_path,
            work_dir,
            bold_skip_frame,
            bold_bandpass
        )
    }

    // --- Step 5: Normalization to template ---
    warp_file_ch = ses_data_flat.map { join_key, subj, ses, ses_key, enhance_dir, resample_dir ->
        def warp = "${enhance_dir}/MEBRAIN/xfms/from-T1w_to-MEBRAIN_mode-image_xfm.nii.gz"
        tuple(ses_key, warp)
    }

    bold_norm_in = bold_preproc_out.combine(bold_anat_flat)
        .filter { sk1, bid, t1wb, jf, sk2, t1w, mask, wm_p, gm_p, csf_p, wm_d, xfm -> sk1 == sk2 }
        .map { sk1, bid, t1wb, jf, sk2, t1w, mask, wm_p, gm_p, csf_p, wm_d, xfm ->
            tuple(sk1, bid, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p)
        }
        .combine(warp_file_ch)
        .filter { sk1, bid, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p,
                  sk2, warp -> sk1 == sk2 }
        .map { sk1, bid, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p,
               sk2, warp ->
            tuple(sk1, bid, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p, warp)
        }

    bold_norm_out = bold_normalize(
        bold_norm_in,
        "${template_space}",
        bold_preprocess_path
    )

    // --- Step 6: Surface projection ---
    surf_ch = ses_data_flat.map { join_key, subj, ses, ses_key, enhance_dir, resample_dir ->
        def ses_prefix = ses ? "_${ses}" : ""
        def subj_ses = "${subj}${ses_prefix}"
        tuple(ses_key,
              "${resample_dir}/ACPC/Native/${subj_ses}_hemi-L_desc-white.surf.gii",
              "${resample_dir}/ACPC/Native/${subj_ses}_hemi-R_desc-white.surf.gii",
              "${resample_dir}/ACPC/Native/${subj_ses}_hemi-L_desc-pial.surf.gii",
              "${resample_dir}/ACPC/Native/${subj_ses}_hemi-R_desc-pial.surf.gii",
              "${resample_dir}/ACPC/Volume/${subj_ses}_desc-ribbon_dseg.nii.gz")
    }

    // Preproc+anat filtered, then combine with surf_ch
    bold_preproc_anat = bold_preproc_out.combine(bold_anat_flat)
        .filter { sk1, bid, t1wb, jf, sk2, t1w, mask, wm_p, gm_p, csf_p, wm_d, xfm -> sk1 == sk2 }
        .map { sk1, bid, t1wb, jf, sk2, t1w, mask, wm_p, gm_p, csf_p, wm_d, xfm ->
            tuple(sk1, bid, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p)
        }

    bold_surf_in = bold_preproc_anat.combine(surf_ch)
        .filter { sk1, bid, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p,
                  sk2, wL, wR, pL, pR, rib -> sk1 == sk2 }
        .map { sk1, bid, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p,
               sk2, wL, wR, pL, pR, rib ->
            tuple(sk1, bid, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p,
                  wL, wR, pL, pR, rib)
        }

    bold_surf_out = bold_vol2surf(
        bold_surf_in,
        hemi_ch,
        bold_preprocess_path
    )

    // --- Step 7: QC ---
    // Build simplified preproc QC channel from bold_preproc_anat
    bold_qc_preproc_in = bold_preproc_anat
        .map { sk, bid, jf, t1w, mask, wm_d, wm_p, gm_p, csf_p ->
            tuple(sk, bid, t1w, bold_preprocess_path)
        }

    // Motion QC
    qc_bold_motion_out = qc_bold_motion(bold_qc_preproc_in, qc_dir)

    // Registration QC
    qc_bold_registration_out = qc_bold_registration(bold_qc_preproc_in, qc_dir)

    // TSNR QC
    qc_bold_tsnr_out = qc_bold_tsnr(bold_qc_preproc_in, qc_dir)

    // Carpet QC
    qc_bold_carpet_out = qc_bold_carpet(bold_qc_preproc_in, qc_dir)

    // Normalization QC: combine bold_norm_out with template-space T1w ref
    // Template-space T1w is in Enhance/MEBRAIN/Volume/
    t1w_template_ch = ses_data_flat.map { join_key, subj, ses, ses_key, enhance_dir, resample_dir ->
        def ses_prefix = ses ? "_${ses}" : ""
        def file_prefix = "${subj}${ses_prefix}"
        tuple(ses_key, "${enhance_dir}/MEBRAIN/${file_prefix}_space-MEBRAIN_desc-restorebrain_T1w.nii.gz")
    }
    bold_qc_norm_in = bold_norm_out.combine(t1w_template_ch)
        .filter { sk1, bid, btemp, sk2, t1w_t -> sk1 == sk2 }
        .map { sk1, bid, btemp, sk2, t1w_t ->
            tuple(sk1, bid, btemp, t1w_t, bold_preprocess_path)
        }
    qc_bold_normalize_out = qc_bold_normalize(bold_qc_norm_in, qc_dir)

    // Surface QC: combine bold_surf_out with surf_ch for ribbon
    bold_qc_surf_in = bold_surf_out.combine(surf_ch)
        .filter { sk1, bid, hemi, sgii, sk2, wL, wR, pL, pR, rib -> sk1 == sk2 }
        .map { sk1, bid, hemi, sgii, sk2, wL, wR, pL, pR, rib ->
            tuple(sk1, bid, hemi, sgii, rib, bold_preprocess_path)
        }
    qc_bold_surf_out = qc_bold_surf(bold_qc_surf_in, qc_dir)

    emit:
    bold_preproc_out
    bold_norm_out
    bold_surf_out
}


process anat_config_init {
    // classify images through file name
    // generate a config file with default parameters
    cpus 1

    input:
      val bids_dir
      val participant_label
      val session_id
      val out_dir

    output:
      val qc_dir

    script:
    qc_dir = "${out_dir}/QC/"

    """
    mkdir -p ${qc_dir}

    params="--data_dir ${bids_dir} --qc_dir ${qc_dir}"

    if [[ -n "${participant_label}" ]]; then
      params="\$params --subj ${participant_label}"
    fi
    
    if [[ -n "${session_id}" ]]; then
      params="\$params --ses ${session_id}"
    fi

    ${params.python_inter} ${params.prepare_script_dir}/scripts/config_all_bids.py \$params
    """
}

// ========================================================================
// Generate QC Summary Report
// ========================================================================
process generate_qc_summary {
    tag "QC Summary - ${ses_key_out}"
    cpus 1
    memory "1GB"

    input:
    val qc_dir
    tuple val(ses_key_out), val(qc_done)

    output:
    val "${qc_dir}/qc_summary.html", emit: qc_report

    script:
    // Split session key into subject_id and session_id
    def (subject_id, session_id) = ses_key_out.split('::')
    def session_qc_dir = "${qc_dir}/${subject_id}/${session_id}"
    def output_html = "${session_qc_dir}/qc_summary.html"

    """
    mkdir -p "${session_qc_dir}"

    ${params.python_inter} ${params.utils_path}/qc_summary.py \\
        --output-dir "${params.out_dir}" \\
        --output "${output_html}" \\
        --qc-dir "${qc_dir}" \\
        --subject "${subject_id}" \\
        --session "${session_id}"

    echo ""
    echo "="*80
    echo "++"
    echo "✅ QC report generated for ${subject_id} / ${session_id}"
    echo "📊 Report: ${output_html}"
    echo "="*80
    """
}

workflow {
    // ========================================================================
    // Parse parameters
    // ========================================================================

    def GPUS = (params.gpus == null) ? inferGpuIndices() : (params.gpus.toString().split(',').collect { it.trim() }.findAll { it })
    def GPU_NUM = GPUS.size() as int

    if (params.debug != 'null') {
        println 'DEBUG: params           : ' + params
    }

    println ''
    println '=' * 80
    println 'MacaSurfer - Multi-Subject Processing (DeepPrep Style)'
    println '=' * 80
    println 'BIDS directory     : ' + params.bids_dir
    println 'Output directory    : ' + params.out_dir
    println 'Participant labels : ' + params.participant_label
    println 'Session labels : ' + params.session_id
    println 'Available GPUs     : ' + GPUS + ' (total ' + GPU_NUM + ')'
    println '=' * 80
    println ''
    
    if (params.debug != 'null') {
        println 'DEBUG: params           : ' + params
    }

    anat_only = params.anat_only.toString().toUpperCase()
    bold_only = params.bold_only.toString().toUpperCase()

    println ''
    println '=' * 80
    println 'MacaSurfer - Multi-Subject Processing (DeepPrep Style)'
    println '=' * 80
    println 'BIDS directory      : ' + params.bids_dir
    println 'Output directory    : ' + params.out_dir
    println 'Participant labels  : ' + params.participant_label
    println 'Session labels      : ' + params.session_id
    println 'Anat only           : ' + anat_only
    println 'BOLD only           : ' + bold_only
    println 'BOLD task type      : ' + params.bold_task_type
    println 'Available GPUs      : ' + GPUS + ' (total ' + GPU_NUM + ')'
    println '=' * 80
    println ''

    if ( "${params.before_check}" == "true" ) {
        qc_dir = anat_config_init(params.bids_dir, params.participant_label, params.session_id, params.out_dir)

    } else if ( bold_only == 'TRUE' ) {
        // ---- BOLD-ONLY mode: skip structural, run BOLD pipeline only ----
        qc_dir = channel.of("${params.out_dir}/QC/")

        // Auto-detect sessions from output directory if not specified
        def subj = "${params.participant_label}"
        def out_root = "${params.out_dir}"
        def session_id_input = params.session_id
        def session_list = []

        if (session_id_input) {
            session_list = [session_id_input]
        } else {
            def subj_dir = new File("${out_root}", subj)
            if (subj_dir.exists() && subj_dir.isDirectory()) {
                subj_dir.listFiles().each { f ->
                    if (f.isDirectory() && f.name.startsWith('ses-')) {
                        session_list.add(f.name)
                    }
                }
            }
            if (!session_list) {
                session_list = ['']  // no sessions found, process without session
            } else {
                println "[INFO] Auto-detected sessions for ${subj}: ${session_list}"
            }
        }

        // Simulate structural pipeline output for bold_wf
        ses_data_in = Channel.fromList(session_list)
            .map { ses ->
                def ses_dir = ses ? "/${ses}" : ""
                def ses_prefix = ses ? "_${ses}" : ""
                def file_prefix = "${subj}${ses_prefix}"
                def ses_key = "${subj}::${ses}"

                tuple(ses_key,
                      [subj], [ses], ['T1'],
                      ["${out_root}/${subj}${ses_dir}/Prepare"],
                      ["${out_root}/${subj}${ses_dir}/Enhance"],
                      ["${out_root}/${subj}${ses_dir}/Surface"],
                      ["${out_root}/${subj}${ses_dir}/Resample"],
                      "${out_root}/${subj}${ses_dir}/Enhance/T1w/${file_prefix}_space-acpc_res-04mm_desc-brain_T1w.nii.gz",
                      "${out_root}/${subj}${ses_dir}/Enhance/T1w/${file_prefix}_space-acpc_res-04mm_desc-brain_T2w.nii.gz",
                      "${out_root}/${subj}${ses_dir}/Enhance/T1w/xfms/acpc.mat",
                      "${out_root}/${subj}${ses_dir}/Enhance/T1w/${file_prefix}_space-acpc_res-04mm_desc-completeaseg_dseg.nii.gz",
                      "${out_root}/${subj}${ses_dir}/Enhance/T1w/${file_prefix}_space-acpc_res-04mm_desc-wmskeleton_mask.nii.gz")
            }
        // Run BOLD workflow
        bold_wf(qc_dir, ses_data_in, Channel.of('bold_only'))
        println ''
        println '=' * 80
        println '[DONE] BOLD-only processing completed!'
        println '=' * 80

    } else if ( "${params.after_check}" == "true" ) {
        qc_dir = channel.of("${params.out_dir}/QC/")
        ses_info = info(qc_dir, params.out_dir)
        prepare_data_out = prepare(qc_dir, ses_info.ctx_keyed, ses_info.run_ctx_keyed, ses_info.run_ctx_by_sess)
        enhance_data_out = enhance(qc_dir, prepare_data_out.avg_out)
        surface_data_out = surface(qc_dir, enhance_data_out.enhance_out)
        resample_data_out = resample(qc_dir, surface_data_out.ses_info, surface_data_out.ses_msmreg_out, surface_data_out.ses_whitesurf_out, surface_data_out.ses_pialsurf_out)
        // Generate QC report after all processing steps
        qc_report = generate_qc_summary(qc_dir, resample_data_out.ses_qc_surface_out)
    } else {
        qc_dir = anat_config_init(params.bids_dir, params.participant_label, params.session_id, params.out_dir)
        ses_info = info(qc_dir, params.out_dir)
        prepare_data_out = prepare(qc_dir, ses_info.ctx_keyed, ses_info.run_ctx_keyed, ses_info.run_ctx_by_sess)
        enhance_data_out = enhance(qc_dir, prepare_data_out.avg_out)
        surface_data_out = surface(qc_dir, enhance_data_out.enhance_out)
        resample_data_out = resample(qc_dir, surface_data_out.ses_info, surface_data_out.ses_msmreg_out, surface_data_out.ses_whitesurf_out, surface_data_out.ses_pialsurf_out)
        // Generate QC report after all processing steps
        qc_report = generate_qc_summary(qc_dir, resample_data_out.ses_qc_surface_out)

        // ---- Run BOLD pipeline after structural (when not anat_only) ----
        if ( anat_only != 'TRUE' ) {
            bold_wf(qc_dir, enhance_data_out, resample_data_out.ses_qc_surface_out)
            println ''
            println '=' * 80
            println '[DONE] Structural + BOLD processing completed!'
            println '=' * 80
        } else {
            println ''
            println '=' * 80
            println '[DONE] Structural-only processing completed!'
            println '=' * 80
        }
    }
}
