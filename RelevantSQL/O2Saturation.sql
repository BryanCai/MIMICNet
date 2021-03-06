SET search_path TO mimiciii;

DROP TABLE public.O2Saturation;
CREATE TABLE public.O2Saturation 
AS (
  SELECT 
    var.subject_id,
    var.icustay_id,
    var.charttime - var.intime,
    var.valuenum AS value
  FROM (
    SELECT
      p.*,
      ce.row_id,
      ce.itemid,
      ce.charttime,
      ce.storetime,
      ce.cgid,
      ce.value,
      ce.valuenum,
      ce.valueuom,
      ce.warning,
      ce.error,
      ce.resultstatus,
      ce.stopped
    FROM pop p 
      LEFT JOIN chartevents ce ON p.icustay_id = ce.icustay_id
  ) var
  WHERE var.itemid IN (220227, 220277)
  AND var.subject_id < 30000
  AND var.charttime BETWEEN var.intime AND var.outtime
);
\COPY public.O2Saturation TO '../Data/O2Saturation.csv' DELIMITER ',' CSV HEADER;