SET search_path TO mimiciii;

-- Size [74152]
DROP TABLE platelet_fusion_ce;
CREATE TABLE platelet_fusion_ce AS (
    SELECT 
        t.subject_id,
        t.icustay_id,
        t.charttime,
        t.value,
        t.stopped
    FROM (
        SELECT 
            p.subject_id,
            p.icustay_id,
            ce.charttime,
            ce.itemid,
            ce.value,
            ce.stopped
        FROM pop p LEFT JOIN chartevents ce 
          ON p.icustay_id = ce.icustay_id
    ) t
    WHERE t.itemid IN (
        828,
        3789,
        227457
    ) AND t.value IS NOT NULL AND t.value <> ''
);

-- Size [1558]
DROP TABLE platelet_fusion_cv;
CREATE TABLE platelet_fusion_cv AS (
    SELECT 
        t.subject_id,
        t.icustay_id,
        t.charttime,
        t.value,
        t.stopped
    FROM (
        SELECT 
            p.subject_id,
            p.icustay_id,
            cv.itemid,
            cv.charttime,
            cv.amount as value,
            cv.stopped
        FROM pop p LEFT JOIN inputevents_cv cv
          ON p.icustay_id = cv.icustay_id
    ) t
    WHERE t.itemid IN (
        30006,
        30105
    ) AND t.value IS NOT NULL
);
ALTER TABLE platelet_fusion_cv ALTER COLUMN value TYPE varchar(255);
ALTER TABLE platelet_fusion_cv ALTER COLUMN stopped TYPE varchar(50);


-- Size [1014]
DROP TABLE platelet_fusion_mv;
CREATE TABLE platelet_fusion_mv AS (
    SELECT 
        t.subject_id,
        t.icustay_id,
        t.charttime,
        t.value,
        t.stopped
    FROM (
        SELECT 
            p.subject_id,
            p.icustay_id,
            mv.itemid,
            mv.endtime AS charttime,
            mv.amount AS value,
            'Stopped' AS stopped
        FROM pop p LEFT JOIN inputevents_mv mv
          ON p.icustay_id = mv.icustay_id
    ) t
    WHERE t.itemid IN (
        225170,
        226369
    ) AND t.value IS NOT NULL
);
ALTER TABLE platelet_fusion_mv ALTER COLUMN value TYPE varchar(255);
ALTER TABLE platelet_fusion_mv ALTER COLUMN stopped TYPE varchar(50);

-- Total Size [76724]
DROP TABLE platelet_fusion;
CREATE TABLE platelet_fusion 
AS (
    SELECT ce.* 
    FROM platelet_fusion_ce ce
    UNION ALL
    SELECT cv.*
    FROM platelet_fusion_cv cv
    UNION ALL 
    SELECT mv.*
    FROM platelet_fusion_mv mv
) ORDER BY subject_id ASC;
       
DROP TABLE platelet_fusion_ce;
DROP TABLE platelet_fusion_cv;
DROP TABLE platelet_fusion_mv;

