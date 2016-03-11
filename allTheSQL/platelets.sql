SET search_path TO mimiciii;

-- Size [216502]
DROP TABLE platelets;
CREATE TABLE platelets 
AS (
  SELECT 
    var.subject_id,
    var.icustay_id,
    var.charttime,
    var.valuenum AS value
  FROM (
    SELECT
      p.*,
      le.row_id,
      le.itemid,
      le.charttime,
      le.value,
      le.valuenum,
      le.valueuom,
      le.flag
    FROM pop p 
      LEFT JOIN labevents le ON p.hadm_id = le.hadm_id
  ) var
  WHERE var.itemid IN (51265)
);
