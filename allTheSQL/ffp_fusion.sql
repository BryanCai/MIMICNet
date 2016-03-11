SET search_path TO mimiciii;

-- Size [3926]
DROP TABLE ffp_fusion_cv;
CREATE TABLE ffp_fusion_cv AS (
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
        30005,
        30103,
        44044,
        44172
    ) AND t.value IS NOT NULL
);
ALTER TABLE ffp_fusion_cv ALTER COLUMN value TYPE varchar(255);
ALTER TABLE ffp_fusion_cv ALTER COLUMN stopped TYPE varchar(50);


-- Size [2131]
DROP TABLE ffp_fusion_mv;
CREATE TABLE ffp_fusion_mv AS (
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
        220970,
        226367,
        227072
    ) AND t.value IS NOT NULL
);
ALTER TABLE ffp_fusion_mv ALTER COLUMN value TYPE varchar(255);
ALTER TABLE ffp_fusion_mv ALTER COLUMN stopped TYPE varchar(50);

-- Total Size [6057]
DROP TABLE ffp_fusion;
CREATE TABLE ffp_fusion 
AS (
    SELECT cv.*
    FROM ffp_fusion_cv cv
    UNION ALL 
    SELECT mv.*
    FROM ffp_fusion_mv mv
) ORDER BY subject_id ASC;
       
DROP TABLE ffp_fusion_cv;
DROP TABLE ffp_fusion_mv;

