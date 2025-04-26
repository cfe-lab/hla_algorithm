LOAD DATA

INFILE '/output/hla_workfinal_dir/hla_a_seq.txt'
BADFILE '/output/hla_workfinal_dir/hla_a_bad.bad'


INTO TABLE specimen.hla_alleles_a

APPEND

(enum TERMINATED BY ',',
alleles_clean TERMINATED BY ',',
alleles_all CHAR(4000) TERMINATED BY ',',
ambiguous TERMINATED BY ',',
homozygous TERMINATED BY ',',
mismatch_count TERMINATED BY ',', 
mismatches CHAR(750)  TERMINATED BY ',',
seq CHAR(787) TERMINATED BY ',',
enterdate SYSDATE)
