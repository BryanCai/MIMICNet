SET search_path TO mimiciii;

DROP TABLE public.pH;
CREATE TABLE public.pH 
AS (
  SELECT 
    var.subject_id,
    var.icustay_id,
    var.charttime,
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
  WHERE var.itemid IN (780, 1126, 4753)
);
\COPY public.pH TO '../Data/pH.csv' DELIMITER ',' CSV HEADER;