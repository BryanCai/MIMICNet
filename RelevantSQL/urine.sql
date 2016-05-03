SET search_path TO mimiciii;

-- Size [697713]
DROP TABLE public.urine;
CREATE TABLE public.urine 
AS (
  SELECT 
    var.subject_id,
    var.icustay_id,
    var.charttime - var.intime,
    var.value
  FROM (
    SELECT
      p.*,
      oe.row_id,
      oe.itemid,
      oe.charttime,
      oe.storetime,
      oe.cgid,
      oe.value,
      oe.valueuom,
      oe.iserror,
      oe.newbottle,
      oe.stopped
    FROM pop p 
      LEFT JOIN outputevents oe ON p.icustay_id = oe.icustay_id
  ) var
  WHERE var.itemid IN (
    -- these are the most frequently occurring urine output observations in CareVue
    40055, -- "Urine Out Foley"
    43175, -- "Urine ."
    40069, -- "Urine Out Void"
    40094, -- "Urine Out Condom Cath"
    40715, -- "Urine Out Suprapubic"
    40473, -- "Urine Out IleoConduit"
    40085, -- "Urine Out Incontinent"
    40057, -- "Urine Out Rt Nephrostomy"
    40056, -- "Urine Out Lt Nephrostomy"
    40405, -- "Urine Out Other"
    40428, -- "Urine Out Straight Cath"
    40096, -- "Urine Out Ureteral Stent #1"
    40651, -- "Urine Out Ureteral Stent #2"

    -- these are the most frequently occurring urine output observations in CareVue
    226559, -- "Foley"
    226560, -- "Void"
    227510, -- "TF Residual"
    226561, -- "Condom Cath"
    227489, -- "GU Irrigant/Urine Volume Out"
    226584, -- "Ileoconduit"
    226563, -- "Suprapubic"
    226564, -- "R Nephrostomy"
    226565, -- "L Nephrostomy"
    226557, -- "R Ureteral Stent"
    226558  -- "L Ureteral Stent"
  )
  AND var.subject_id < 30000
  AND var.charttime BETWEEN var.intime AND var.outtime
);
\COPY public.urine TO '../Data/urine.csv' DELIMITER ',' CSV HEADER;