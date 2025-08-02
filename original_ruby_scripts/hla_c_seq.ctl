LOAD DATA

INFILE '/output/hla_workfinal_dir/hla_c_seq.txt'
BADFILE '/output/hla_workfinal_dir/hla_c_bad.bad'


INTO TABLE specimen.hla_alleles_c

APPEND

(enum TERMINATED BY ',',
alleles_clean  TERMINATED BY ',',
alleles_all CHAR(1500) TERMINATED BY ',',
ambiguous  TERMINATED BY ',',
homozygous  TERMINATED BY ',',
mismatch_count TERMINATED BY ',', 
mismatches CHAR(750) TERMINATED BY ',',
seqa CHAR(270)  TERMINATED BY ',',
seqb CHAR(276) TERMINATED BY ',',
enterdate SYSDATE)
