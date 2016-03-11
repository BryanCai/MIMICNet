SET search_path TO mimiciii;

-- Size [26052]
DROP TABLE weaning;
CREATE TABLE weaning 
AS (
    SELECT 
        t.*
    FROM (
        SELECT 
            part3.*,
            sofa.sofa
        FROM (
            SELECT 
                part2.*,
                oasis.oasis
            FROM (
                SELECT 
                    part1.*,
                    saps.saps
                FROM (
                    SELECT 
                        p.subject_id,
                        p.icustay_id,
                        p.intime,
                        p.outtime,
                        p.hospital_expire_flag,
                        p.dod,
                        p.age,
                        p.gender,
                        hw.weight_first,
                        hw.height_first, 
                        hw.weight_first / (hw.height_first * hw.height_first) AS bmi,
                        p.first_careunit,
                        p.pacemaker,
                        p.riskfalls
                    FROM pop p LEFT JOIN heightweight hw ON p.icustay_id=hw.icustay_id 
                    WHERE p.icustay_seq = 1
                ) part1 LEFT JOIN saps ON part1.icustay_id=saps.icustay_id
            ) part2 LEFT JOIN oasis ON part2.icustay_id=oasis.icustay_id
        ) part3 LEFT JOIN sofa ON part3.icustay_id=sofa.icustay_id
    ) t 
);