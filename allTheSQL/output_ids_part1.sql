SELECT d_items.label, MIN(d_items.itemid) item_id, count(distinct(hadm_id)) hadm_id_count
FROM  mimiciii.d_items, mimiciii.chartevents
WHERE chartevents.itemid = d_items.itemid
AND (
upper(label) like '%DOBUTAMINE%'
or upper(label) like '%DOPAMINE%'
or upper(label) like '%EPINEPHRINE%'
or upper(label) like '%LEVOPHED%'
or upper(label) like '%VASOPRESSIN%'
or upper(label) like '%MILRINONE%'
or upper(label) like '%NEOSYNEPHRINE%')
group by d_items.label
ORDER BY label ASC 
;