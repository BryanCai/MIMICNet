SET search_path TO mimiciii;

-- Size [697713]
DROP TABLE public.codes;
CREATE TABLE public.codes 
AS (
  SELECT 
    var.subject_id,
    var.icustay_id,
    var.charttime,
    var.icd9_code
  FROM (
    SELECT
      p.*,
      dia.row_id,
      dia.itemid,
      dia.seq_num
      dia.icd9_code
    FROM pop p 
      LEFT JOIN diagnoses_icd dia ON p.hadm_id = dia.hadm_id
  ) var
  WHERE var.seq_num = 1
);
\COPY public.codes TO '../Data/codes.csv' DELIMITER ',' CSV HEADER;