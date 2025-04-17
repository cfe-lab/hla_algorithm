=begin

03-08-2023 wscott add optparse, code cleanup for docker container
03-06-2024 wscott make sure that the HLA_WORKTEXT and HLA_WORKFINAL
                  directories are empty before we process samples to avoid
                  data pollution..
                  Clean up code (remove unused variables), add more options
                  for --verbose flag
=end

require 'open3'
require 'stringio'
require 'fileutils'
require 'optparse'

require 'ckwlib/login'
require 'ckwlib/cfe_env'
# require 'ckwlib/ckw'
require 'ckwlib/hla_algorithm'
require 'ckwlib/sequence'
# require 'ckwlib/report_encrypt_manager'

PROG_NAME = 'hla.rb'.freeze
VERSION_STRING = '1.0'.freeze
VERSION_DATE = '06-Jun-2024'.freeze


# 03-08-2023 wscott
# The outline of the program is as follows:
# 1) Files in the uploaddir are scanned.
# Copies of these files are written to HLA_WORKTEXT_DIR in
#  modified form, and filenames are changed to a canonical
# form if required. The files in uploaddir are not modified or deleted.
#
# 2) The files in HLA_WORKTEXT_DIR are scanned. The sequence sizes
# are checked and any with a wrong size is discarded.
# Here, the sequences with some other information is held
# in the hla_X_processing arrays.
#
# 3) for each letter a,b,c, the "hla_#{letter}_std_reduced.csv" is read in
# and each sequence in hla_X_processing compared to these sequences.
# the matching information is appended to the hla_X_processing entries.
#
# 4) the matches found are then 'cleaned' which removes some ambiguity...
#
# 5) Homozygosity is then searched for in the hla_X_processing arrays
#
# 6) for the hla_b_processing arrays only, the code looks for B*5701 allelles.
#
# 7) The contents of the hla_X_processing arrays are written to
# HLA_WORKFINAL_DIR/hla_X_seq.txt
# This information contains the Allele classification of all scanned sequences.




# Because the code will live in on a read-only file system,
# we have to differentiate between these cases:
# HLA_PROG_DIR: the place where code is located (ro)
# HLA_CONF_DIR: the place where hla configuration files are read from (ro)
# These are the HLA reference sequences: hla_X_std_reduced.csv
# and the hla_frequencies.csv file.

# HLA_WORKTEXT_DIR: an intermediate work directory (rw)
# HLA_WORKFINAL_DIR (rw)
#   an intermediate work directory
#  the place into which logfiles are written (rw)

# HLA_PROCESSED_DIR: the place into which
# text (sequences from the HLA_WORKTEXT_DIR) and final(classificaion from HLA_WORKFINAL_DIR)
# files are written into after database upload (rw)

# cons_b5701 = 'GCTCCCACTCCATGAGGTATTTCTACACCGCCATGTCCCGGCCCGGCCGCGGGGAGCCCCGCTTCATCGCAGTGGGCTACGTGGACGACACCCAGTTCGTGAGGTTCGACAGCGACGCCGCGAGTCCGAGGATGGCGCCCCGGGCGCCATGGATAGAGCAGGAGGGGCCGGAGTATTGGGACGGGGAGACACGGAACATGAAGGCCTCCGCGCAGACTTACCGAGAGAACCTGCGGATCGCGCTCCGCTACTACAACCAGAGCGAGGCCG,GGTCTCACATCATCCAGGTGATGTATGGCTGCGACGTGGGGCCGGACGGGCGCCTCCTCCGCGGGCATGACCAGTCYGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGAGCTCCTGGACCGCGGCGGACACGGCGGCTCAGATCACCCAGCGCAAGTGGGAGGCGGCCCGTGTGGCGGAGCAGCTGAGAGCCTACCTGGAGGGCCTGTGYGTGGAGTGGCTCCGCAGATACCTGGAGAACGGGAAGGAGACGCTGCAGCGCGCGG'

B570101 = 'GCTCCCACTCCATGAGGTATTTCTACACCGCCATGTCCCGGCCCGGCCGCGGGGAGCCCCGCTTCATCGCAGTGGGCTACGTGGACGACACCCAGTTCGTGAGGTTCGACAGCGACGCCGCGAGTCCGAGGATGGCGCCCCGGGCGCCATGGATAGAGCAGGAGGGGCCGGAGTATTGGGACGGGGAGACACGGAACATGAAGGCCTCCGCGCAGACTTACCGAGAGAACCTGCGGATCGCGCTCCGCTACTACAACCAGAGCGAGGCCG,GGTCTCACATCATCCAGGTGATGTATGGCTGCGACGTGGGGCCGGACGGGCGCCTCCTCCGCGGGCATGACCAGTCCGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGAGCTCCTGGACCGCGGCGGACACGGCGGCTCAGATCACCCAGCGCAAGTGGGAGGCGGCCCGTGTGGCGGAGCAGCTGAGAGCCTACCTGGAGGGCCTGTGCGTGGAGTGGCTCCGCAGATACCTGGAGAACGGGAAGGAGACGCTGCAGCGCGCGG'.freeze

B570102 = 'GCTCCCACTCCATGAGGTATTTCTACACCGCCATGTCCCGGCCCGGCCGCGGGGAGCCCCGCTTCATCGCAGTGGGCTACGTGGACGACACCCAGTTCGTGAGGTTCGACAGCGACGCCGCGAGTCCGAGGATGGCGCCCCGGGCGCCATGGATAGAGCAGGAGGGGCCGGAGTATTGGGACGGGGAGACACGGAACATGAAGGCCTCCGCGCAGACTTACCGAGAGAACCTGCGGATCGCGCTCCGCTACTACAACCAGAGCGAGGCCG,GGTCTCACATCATCCAGGTGATGTATGGCTGCGACGTGGGGCCGGACGGGCGCCTCCTCCGCGGGCATGACCAGTCTGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGAGCTCCTGGACCGCGGCGGACACGGCGGCTCAGATCACCCAGCGCAAGTGGGAGGCGGCCCGTGTGGCGGAGCAGCTGAGAGCCTACCTGGAGGGCCTGTGCGTGGAGTGGCTCCGCAGATACCTGGAGAACGGGAAGGAGACGCTGCAGCGCGCGG'.freeze

B570103 = 'GCTCCCACTCCATGAGGTATTTCTACACCGCCATGTCCCGGCCCGGCCGCGGGGAGCCCCGCTTCATCGCAGTGGGCTACGTGGACGACACCCAGTTCGTGAGGTTCGACAGCGACGCCGCGAGTCCGAGGATGGCGCCCCGGGCGCCATGGATAGAGCAGGAGGGGCCGGAGTATTGGGACGGGGAGACACGGAACATGAAGGCCTCCGCGCAGACTTACCGAGAGAACCTGCGGATCGCGCTCCGCTACTACAACCAGAGCGAGGCCG,GGTCTCACATCATCCAGGTGATGTATGGCTGCGACGTGGGGCCGGACGGGCGCCTCCTCCGCGGGCATGACCAGTCCGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGAGCTCCTGGACCGCGGCGGACACGGCGGCTCAGATCACCCAGCGCAAGTGGGAGGCGGCCCGTGTGGCGGAGCAGCTGAGAGCCTACCTGGAGGGCCTGTGTGTGGAGTGGCTCCGCAGATACCTGGAGAACGGGAAGGAGACGCTGCAGCGCGCGG'.freeze


# ------ script starts here ----------------
=begin
 NOTE: this code is now in ckwlib/sequence.rb
REV_AMBIG = AMBIG.invert
SUPER_COMBO = Hash.new
arr = ['A','C','G','T']
arr_combo = []
arr.each do |a|
  arr.each do |b|
    arr_combo << [a,b]
  end
end

arr_combo.each do |a|
  arr_combo.each do |b|
    arr_combo.each do |c|
      SUPER_COMBO[[a[0] + b[0] + c[0], a[1] + b[1] + c[1]]] = REV_AMBIG[a.uniq.sort] + REV_AMBIG[b.uniq.sort] + REV_AMBIG[c.uniq.sort]
    end
  end
end
=end

# CODE and config files are all relative to LAB_UPLOAD_ROOT
# and will typically all live in the docker container
LAB_UPLOAD_ROOT = $cfe_env[:lab_upload_root]
raise 'LAB_UPLOAD_ROOT is not defined' if LAB_UPLOAD_ROOT.nil?
HLA_PROG_DIR = File.join(LAB_UPLOAD_ROOT, 'hla')

HLA_GEN_REPORTS_SCRIPT = File.join(HLA_PROG_DIR, '_generate_hla_reports.rb')


HLA_CONF_DIR = File.join(HLA_PROG_DIR, 'config')

HLA_CONF_FREQUENCY_FILE = File.join(HLA_CONF_DIR,
                                    'hla_frequencies.csv')


# INPUT DIRECTORY
sanger_runs_dir = $cfe_env[:output_sanger_runs_dest]
raise 'output_sanger_runs_dest cfe_env entry is not defined!' if sanger_runs_dir.nil?
DEFAULT_UPLOAD_DIR = File.join(sanger_runs_dir, 'HLA_TXT_SEQUENCES')

# WORK directories
HLA_WORKTEXT_DIR = $cfe_env[:output_hla_worktext_dir]
raise 'HLA_WORKTEXT_DIR is not defined' if HLA_WORKTEXT_DIR.nil?

HLA_WORKFINAL_DIR = $cfe_env[:output_hla_workfinal_dir]
raise 'HLA_WORKFINAL_DIR is not defined' if HLA_WORKFINAL_DIR.nil?
# this log file is created here at run-time, but then moved to
# HLA_LOG_DIR at the end.
HLA_TEMP_LOG_FILE=File.join(HLA_WORKFINAL_DIR, 'log.txt')

HLA_LOG_DIR = $cfe_env[:output_hla_log_dir]
raise 'HLA_LOG_DIR is not defined' if HLA_LOG_DIR.nil?

# The OUTPUT PROCESSED DIRECTORY
HLA_PROCESSED_DIR = $cfe_env[:output_hla_processed_dir]
raise 'HLA_PROCESSED_DIR is not defined' if HLA_PROCESSED_DIR.nil?

# SQLLDR executable...
SQLLDR_EXECUTABLE = $cfe_env[:sqlldr]
raise 'SQLLDR_EXECUTABLE is not defined' if SQLLDR_EXECUTABLE.nil?

HLA_STD_REDUCED_PATHS = find_hla_std_paths()


# perform a configuration test (make sure all files and
# directories needed are defined )
# Return a boolean: 'the config test passed'
def config_test(upload_dir)
  #
  # HLA_CONF_DIR must exist with the appropriate files.
  # these files are readonly
  conf_files = [upload_dir,
                HLA_CONF_DIR,
                HLA_CONF_FREQUENCY_FILE,
                HLA_GEN_REPORTS_SCRIPT,
                HLA_STD_REDUCED_PATHS['a'],
                HLA_STD_REDUCED_PATHS['b'],
                HLA_STD_REDUCED_PATHS['c']]
  all_ok = true
  conf_files.each do |fcheck|
    unless File.exist?(fcheck)
      all_ok = false
      puts "required file or directory #{fcheck} is missing"
    end
  end
  # these directories must exist and be writable..
  wr_dirs = [HLA_WORKTEXT_DIR,
             HLA_WORKFINAL_DIR,
             HLA_LOG_DIR,
             HLA_PROCESSED_DIR]
  wr_dirs.each do |dcheck|
    if File.exist?(dcheck)
      unless File.writable?(dcheck)
        puts "directory #{dcheck} is not writable"
        all_ok = false
      end
    else
      all_ok = false
      puts "required directory #{dcheck} is missing"
    end
  end
  return all_ok
end


# these are the old command line arguments
# username, password = *(ARGV[0].split('/')) if(ARGV[0])
# autodir = ARGV[1] if(ARGV[1])
# no_db_upload = ARGV[2] == 'nodb'
# if(username == nil or username == '')
#  login_hash = {}
#  login(login_hash) { |dbh| }
#  username = login_hash[:username]
#  password = login_hash[:password]
# end
options = {}
options[:verbose] = false
options[:configtest] = false
options[:login] = nil
options[:automode] = false
options[:usedb] = true
options[:scandir] = nil
puts "#{PROG_NAME} version #{VERSION_STRING} (#{VERSION_DATE})"
OptionParser.new do |opts|
  opts.banner = "Usage: #{PROG_NAME} [options] upload_dir"
  opts.on('-l', '--login OPT', 'Login (user/pass)') do |v|
    options[:login] = v
  end
  opts.on('--nodb', 'Do not access the database') do
    options[:usedb] = false
  end
  opts.on('--automode', 'Do not perform user interaction') do
    options[:automode] = true
  end
  opts.on('--scandir OPT', "Scan OPT directory for input HLA *.txt files (default #{DEFAULT_UPLOAD_DIR})") do |v|
    options[:scandir] = v
  end
  # opts.on('-a', '--anon', 'Anonymise?') do |v|
  #  options[:anon] = v
  # end
  # opts.on('-d', '--devel', 'developer mode, do not send emails') do
  #  options[:devel] = true
  # end
  opts.on('--configtest', 'Test the general configuration and exit.') do
    options[:configtest] = true
  end
  opts.on('-v', '--verbose', 'Be verbose when running?') do |v|
    options[:verbose] = v
  end
end.parse!

# determine the directory to scan for files...
uploaddir = DEFAULT_UPLOAD_DIR
# check for optional upload directory at end of arglist..
unless ARGV.empty?
  if ARGV.size != 1
    puts "exactly one upload directory expected, but got #{ARGV}"
    exit(1)
  end
  if options[:scandir]
    puts 'cannot provide two upload directories! (one via --scandir, one via positional argument)'
    exit(1)
  end
  uploaddir = ARGV[0]
else
  uploaddir = options[:scandir] if options[:scandir]
end

puts "uploaddir is #{uploaddir}"

if options[:automode] and (!options[:usedb])
  puts 'Running in automode requires database access'
  exit(1)
end
if options[:login].nil?
  if options[:usedb]
    puts 'Running with database access, but no password provided. Exiting'
    exit(1)
  end
end

if options[:configtest]
  config_ok = config_test(uploaddir)
  if config_ok
    puts 'configtest passed'
    exit(0)
  else
    puts 'configtest failed'
    exit(1)
  end
end

# determine login hash. Leave this empty if no -l option was provided
login_hash = {}
if options[:login]
  username, password = *(options[:login] and options[:login].split('/').size == 2 ? options[:login].split('/') : [nil,nil])
  if username.nil? or password.nil?
    puts 'format error for username/password option provided with -l option'
    exit(1)
  end
  login_hash[:username] = username
  login_hash[:password] = password
end

automode = options[:automode]

# -- now start the actual work....
config_ok = config_test(uploaddir)
if config_ok
  puts 'configtest passed'
else
  puts 'configtest failed'
end
raise 'configtest failed' unless config_ok

# make sure that HLA_WORKTEXT_DIR is empty before we start...
begin
  wkfiles = Dir[File.join(HLA_WORKTEXT_DIR, "*.TXT")]
  wkfiles.each do |delfile|
    puts "cleaning up work text #{wkfiles}" if options[:verbose]
    FileUtils.rm(delfile)
  end
  # make sure that HLA_WORKFINAL_DIR is empty before we start...
  wkfiles = Dir[File.join(HLA_WORKFINAL_DIR, "*")]
  wkfiles.each do |delfile|
    puts "cleaning up work final #{wkfiles}" if options[:verbose]
    FileUtils.rm(delfile)
  end
end

puts "Scanning directory #{uploaddir}"
# check for uploaddir here...

FileUtils.cd(HLA_PROG_DIR)


#---- fix any funny looking files.
afiles = Dir[File.join(uploaddir, "*-A.TXT")] + Dir[File.join(uploaddir, "*.A.TXT")]
afiles.each do |file|
  #	FileUtils.mv(file, file.gsub(/\-/, '_'))
  file =~ /\/([^\/]+)[\_\-\.]A\.TXT/i
  newfilename = $1 + "_A.TXT"
  File.open(File.join(HLA_WORKTEXT_DIR, newfilename), 'w') do |outfile|
    File.open(file) do |infile|
      outfile.print infile.gets(nil).strip.gsub(/^>.+\n/,'').gsub(/\s/, '')
    end
  end
end

bfiles = Dir.glob(File.join(uploaddir, "*-B?.TXT"), File::FNM_CASEFOLD) +
         Dir.glob(File.join(uploaddir, "*_B?.TXT"), File::FNM_CASEFOLD) +
         Dir.glob(File.join(uploaddir, "*.B?.TXT"), File::FNM_CASEFOLD)
bfiles = bfiles.partition {|a| a =~ /[aA]\.TXT/i }.map {|l| l.sort }

0.upto(bfiles[0].size - 1) do |i|
  bfiles[0][i] =~ /\/([^\/]+)[\_\-\.]Ba\.TXT/i
  newfilename = $1 + "_B.TXT"
  bfiles[1][i] =~ /\/([^\/]+)[\_\-\.]Bb\.TXT/i
  if (newfilename != $1 + "_B.TXT")
    puts "Error: #{bfiles[0][i]} and #{bfiles[1][i]} do not match, gonna quit & die"
    # gets if(!autodir)
    exit(2)
  end
  File.open(File.join(HLA_WORKTEXT_DIR, newfilename), 'w') do |outfile|
    File.open(bfiles[0][i]) do |infile|
      outfile.print infile.gets(nil).strip.gsub(/^>.+\n/,'').gsub(/\s/, '')
    end
    outfile.print ','
    File.open(bfiles[1][i]) do |infile|
      outfile.print infile.gets(nil).strip.gsub(/^>.+\n/,'').gsub(/\s/, '')
    end
  end
end

cfiles = Dir[File.join(uploaddir, "*-C?.TXT")] +
         Dir[File.join(uploaddir, "*_C?.TXT")] +
         Dir[File.join(uploaddir, "*.C?.TXT")]
cfiles = cfiles.partition {|a| a =~ /[aA]\.TXT/i }.map {|l| l.sort }
0.upto(cfiles[0].size - 1) do |i|
  cfiles[0][i] =~ /\/([^\/]+)[\_\-\.]Ca\.TXT/i
  newfilename = $1 + "_C.TXT"
  cfiles[1][i] =~ /\/([^\/]+)[\_\-\.]Cb\.TXT/i
  if (newfilename != $1 + "_C.TXT")
    puts "Error: #{cfiles[0][i]} and #{cfiles[1][i]} do not match, gonna quit & die"
    # gets if(!autodir)
    exit(2)
  end
  File.open(File.join(HLA_WORKTEXT_DIR, newfilename), 'w') do |outfile|
    File.open(cfiles[0][i]) do |infile|
      outfile.print infile.gets(nil).strip.gsub(/^>.+\n/,'').gsub(/\s/, '')
    end
    outfile.print ','
    File.open(cfiles[1][i]) do |infile|
      outfile.print infile.gets(nil).strip.gsub(/^>.+\n/,'').gsub(/\s/, '')
    end
  end
end
#---------------

log = StringIO.new('', 'w')

# MUST ADD LOGGING
begin
  # Automatically process everything in the source folder
  log.puts 'Processing HLA sequences'
  source_files = Dir[File.join(HLA_WORKTEXT_DIR, '*')]

  hla_a_source_files = source_files.find_all {|f| f.include?('_A.txt') or f.include?('_A.TXT')}
  hla_b_source_files = source_files.find_all {|f| f.include?('_B.txt') or f.include?('_B.TXT')}
  hla_c_source_files = source_files.find_all {|f| f.include?('_C.txt') or f.include?('_C.TXT')}

  # Processing files look like ['filename', 'sampleid', 'full_text', 'cutdown text']
  hla_a_processing = []
  hla_b_processing = []
  hla_c_processing = []

  # check file sizes for each of the sequences.
  hla_a_source_files.each do |fname|
    fname =~ /#{HLA_WORKTEXT_DIR}\/(.+)[_-]A\.txt/i
    samp = $1
    text = ''
    File.open(fname) do |fin|
      text = fin.gets(nil)
      if (text.size != 787)
	log.puts "#{samp} HLA A's sequence is the wrong size, expected 547 characters but found #{text.size}"
	log.puts "Skipping sequence"
	next
      end
      if (!(text =~ /^[atgcrykmswnbdhv]{787,787}$/i))
	log.puts "#{samp} HLA A's sequence has invalid characters."
	log.puts "Skipping sequence"
	next
      end
    end
    hla_a_processing.push([fname, samp,
                           text, text[0 .. 270] + text[512 .. 787]])
  end

  hla_b_source_files.each do |fname|
    fname =~ /#{HLA_WORKTEXT_DIR}\/(.+)_B\.txt/i
    samp = $1
    text = ''
    File.open(fname) do |fin|
      text = fin.gets(nil)
      text_pieces = text.split(',')
      if (text_pieces[0].size != 270)
	log.puts "#{samp} HLA B's segment 1 sequence is the wrong size, expected 270 characters but found #{text_pieces[0].size}"
	log.puts "Skipping sequence"
	next
      end
      if (text_pieces[1].size != 276)
	log.puts "#{samp} HLA B's segment 2 sequence is the wrong size, expected 276 characters but found #{text_pieces[1].size}"
	log.puts "Skipping sequence"
	next
      end
      if (!(text =~ /^[atgcrykmswnbdhv]{270,270},[atgcrykmswnbdhv]{276,276}$/i))
	log.puts "#{samp} HLA B's sequence has invalid characters."
	log.puts "Skipping sequence"
	next
      end
    end
    hla_b_processing.push([fname, samp, text, text.gsub(',','')])
  end

  hla_c_source_files.each do |fname|
    fname =~ /#{HLA_WORKTEXT_DIR}\/(.+)_C\.txt/i
    samp = $1
    text = ''
    File.open(fname) do |fin|
      text = fin.gets(nil)
      text_pieces = text.split(',')
      if (text_pieces[0].size != 270)
	log.puts "#{samp} HLA C's segment 1 sequence is the wrong size, expected 270 characters but found #{text_pieces[0].size}"
	log.puts 'Skipping sequence'
	next
      end
      if (text_pieces[1].size != 276)
	log.puts "#{samp} HLA C's segment 2 sequence is the wrong size, expected 276 characters but found #{text_pieces[1].size}"
	log.puts 'Skipping sequence'
	next
      end
      if (!(text =~ /^[atgcrykmswnbdhv]{270,270},[atgcrykmswnbdhv]{276,276}$/i))
	log.puts "#{samp} HLA C's sequence has invalid characters."
	log.puts 'Skipping sequence'
	next
      end
    end
    hla_c_processing.push([fname, samp, text, text.gsub(',','')])
  end

  # exit here without an error if we have nothing to process...
  if hla_a_processing.empty? && hla_b_processing.empty? && hla_c_processing.empty?
    puts 'no entries to process found, now exiting'
    exit(0)
  end

# Previous code
=begin
hla_freqs = []
File.open("#{HLA_DIR}hla_frequencies.csv") do |file|
	file.each_line do |line|
		tmp = line.split(',').map{|a| a.strip }
    hla_freqs << tmp.map {|a| [a[0,2], a[2,2]] }
	end
end
=end

# Rosemary's optimization
  #=begin
  puts "loading frequency file #{HLA_CONF_FREQUENCY_FILE}"
  hla_freqs = [{},{},{}]
  File.open(HLA_CONF_FREQUENCY_FILE) do |file|
    file.each_line do |line|
      row = line.split(',').map{|a| a.strip }
      0.upto(2) do |column|
        tmp = row[column*2, 2].map {|a| [a[0,2], a[2,2]] }
        hla_freqs[column][tmp] = 0 if hla_freqs[column][tmp] == nil
        hla_freqs[column][tmp] += 1
      end
    end
  end
#=end

  # Fun, this is nice repeatable useful code!
  [[hla_a_processing, 'a'],
   [hla_b_processing, 'b'],
   [hla_c_processing, 'c']].each do |processing|
    hla_processing = processing[0]
    letter = processing[1]
    # sco hla_consensus = []
    hla_stds = []
    # hla_vars = []
    GC.start

    puts "Processing HLA #{letter.upcase} Classification"
    puts "#{hla_processing.size()} sample entries to process..."
    if hla_processing.size > 0
      std_file = HLA_STD_REDUCED_PATHS[letter]
      puts "   reading standard (reduced) file #{std_file}"
      File.open(std_file) do |file|
        file.each_line do |line|
          tmp = line.strip.split(',')
          # seq = ''
          # hla_vars.each_with_index {|v, i| seq += tmp[0][v, 1] }
          hla_stds.push([tmp[0], tmp[1] + tmp[2]])
        end
      end
      puts "   read #{hla_stds.size()} lines"
    end
    hla_processing.each do |entry|
      puts "Processing sample #{entry[1]}";
      # sco hla_consensus = []
      # sco entry_seq_reduced = ''
      # hla_vars.each do |v|
      #    entry_seq_reduced += entry[3][v, 1]
      # end
      entry_seq_reduced = entry[3]
      # [allele, seq]
      matching_stds = []
      # sco hla_processing = [] # Pointless?
      tim = Time.now.to_i
      puts '   Building std List'
      hla_stds.each do |std|
        mismatch_cnt = std_match(std[1], entry[3])
        if (mismatch_cnt < 5)
          #if(std_match(std[1], entry[3]) < 10)
          matching_stds.push([std, mismatch_cnt])
        end
      end
      puts "   Done in #{Time.now.to_i - tim} seconds"
      puts "   #{matching_stds.size} standards with less than 6 mismatches "
      puts '   Combining standards'
      tim = Time.now.to_i
      # TODO
      # Now, combine all the stds (pick up that can citizen!)
      # NOTE (wscott): we are actually updating the min structure
      # in which min[0] is the currently smallest sequence distance encountered.
      min = [9999, []]
      alleles_hash = Hash.new
      # alleles_hash.default = []
      # This is the slow part.
      # Rosemary's optimizations have been added
      matching_stds.each_with_index do |std_ax, i_a|
        std_a, std_a_min = std_ax[0], std_ax[1]
        next if(std_a_min > min[0])
        matching_stds.each_with_index do |std_bx, i_b|
          next if (i_b < i_a) # I think this makes sense...
          std_b, std_b_min = std_bx[0], std_bx[1]
          next if (std_b_min > min[0])
          # The key to Rosemary's optimizations
          std = ''
          # SUPER_COMBO
          0.upto((std_a[1].size / 3) - 1) do |i|
            # This bit really takes a while.  How to improve...
            std += SUPER_COMBO[[std_a[1][i * 3,3], std_b[1][i * 3,3]]]
          end

          combistr = "#{std_a[0]} - #{std_b[0]}"
          alleles_hash[std] = [] if (alleles_hash[std] == nil)
          alleles_hash[std].push(combistr)
          # if options[:verbose]
          #  puts "found #{i_a}:#{i_b} #{combistr} #{alleles_hash[std].size}"
          # end
          # check to see if I need to add it to min
          if (std == entry_seq_reduced)
            # Look for exact matches first
            if (min[0] != 0)
              min[0] = 0
              min[1] = [[std, alleles_hash[std]]]
            elsif (!min[1].include?([std, alleles_hash[std]]))
              min[1].push([std, alleles_hash[std]])
              # The element should get continuously updated, right?
            end
            next
          end

          mismatches = 0
          0.upto(std.size() - 1) do |dex|
	    if (std[dex, 1] != entry_seq_reduced[dex, 1])
	      mismatches += 1
	    end
	    break if (mismatches > min[0])
	  end

	  if (mismatches == min[0] and !min[1].include?([std, alleles_hash[std]]))
	    min[1].push([std, alleles_hash[std]])
	  elsif (mismatches < min[0])
	    min[0] = mismatches
	    min[1] = [[std, alleles_hash[std]]]
	  end
        end
      end
      # alleles_hash.each do |std, allele_list|
      #    hla_consensus.push([std, allele_list])
      # end
      puts "   Done in #{Time.now.to_i - tim} seconds"
      # 2024-06-04 this is always zero
      # puts "   Stds combined, now have #{hla_consensus.size} standards"
      puts "   MIN found: #{min[1]}" if options[:verbose]
      # Hrm......
      # puts "rest of it"
      tim = Time.now.to_i
      # Get list of mismatches
      mislist = []
      puts "   Found a match with #{min[0]} mismatches"
      min[1].each do |cons|  # really, we should just pick one, right?
        0.upto(cons[0].size - 1) do |dex|
          if (cons[0][dex, 1] != entry_seq_reduced[dex, 1])
	    mislist.push("#{((dex + ((letter == 'a' and dex > 270) ? 241 : 0)) + 1).to_s}:#{entry[3][dex, 1]}")
	  end
        end
        break
      end
      mislist.uniq!
      puts "   mismatch list: #{mislist.join(', ')}"
      entry.push([min[0], mislist.join(';')]) # mismatches_count, mismatches.
      entry.push(min[1])
    end # hla_processing.each do |entry|
  end
  # [[hla_a_processing, 'a'], [hla_b_processing, 'b'], [hla_c_processing, 'c']].each do |processing|

  #---Probably stop here?

  # Clean the alleles
  puts 'Cleaning Alleles'
  [[hla_a_processing, 'a', 'A\*'],
   [hla_b_processing, 'b', 'B\*'],
   [hla_c_processing, 'c', 'C\*']].each do |processing|
    prefix = processing[2]
    fcnt = 0 # used to offset the freq column
    fcnt = 2 if (processing[1] == 'b')
    fcnt = 4 if (processing[1] == 'c')
    processing[0].each do |entry|
      clean_allele = ''
      alleles = []
      ambig = '0'

      entry[5].each do |a|
        alleles += a[1]
      end
      puts "   cleaning #{entry[1]}"

      # Need a new way to resolve ambiguities right here.
      # Must take into account BOTH alleles to find the most common
      collection = alleles.map do |a|
        tmp = a.split('-')
        # [tmp[0].gsub(/[^\d]/, '')[0, 8], tmp[1].gsub(/[^\d]/, '')[0, 8]]
        [tmp[0].gsub(/[^\d:]/, '').split(':'), tmp[1].gsub(/[^\d:]/, '').split(':')]
      end
      # puts collection.inspect
      # puts "FIrst"
      # puts alleles.map{|a| "|#{a}|"}
      if (collection.map{|e| [e[0][0], e[1][0]]}.uniq.size != 1)
        ambig = '1'
        collection_ambig = collection.map{|e| [e[0][0 .. 1], e[1][0 .. 1]]}.uniq
        # previous code
        #      collection_ambig.each do |a|
        #        a.push(hla_freqs.find_all {|e| e[0 + fcnt,2] == a }.size)
        #      end
        # Rosemary's code
        collection_ambig.each do |a|
          freq = hla_freqs[fcnt/2][a]
          if (freq == nil)
            freq = 0
          end
          a.push(freq)
        end
        max_allele = collection_ambig.max do |a,b|
          if (a[2] != b[2]) # Go by frequency
            a[2] <=> b[2]
          elsif (b[0][0].to_i != a[0][0].to_i) # Then lowest first allele
            b[0][0].to_i <=> a[0][0].to_i
          elsif (b[0][1].to_i != a[0][1].to_i)
            b[0][1].to_i <=> a[0][1].to_i
          elsif (b[1][0].to_i != a[1][0].to_i) # Then lowest second allele
            b[1][0].to_i <=> a[1][0].to_i
          else
            b[1][1].to_i <=> a[1][1].to_i
          end
        end
        a1 = max_allele[0][0]
        a2 = max_allele[1][0]
        #            puts "Max"
        #            puts "(" + a1 + " " + a2 + ")"
        #            alleles.delete_if {|a| !(a =~ /^#{prefix}#{a1}([^\s])+\s-\s#{prefix}#{a2}/)}
        alleles.delete_if {|a| !(a =~ /^#{prefix}#{a1}:([^\s])+\s-\s#{prefix}#{a2}:/)}
      end
      #        puts "Now"
      #        puts alleles
      # non ambiguous now, do the easy way
      # OK, make sure all the alleles mostly match
      collectiona = alleles.map do |a|
        tmp = a.split('-')
        tmp[0].gsub!(/[^\d:]/, '')
        tmp[0].split(':')
      end

      # puts collectiona.inspect
      0.upto(0) do
        if (collectiona.map{|a| a[0, 4]}.uniq.size == 1)
	  clean_allele += prefix + collectiona[0][0,4].join(':') + " - "
        elsif (collectiona.map{|a| a[0, 3]}.uniq.size == 1)
	  clean_allele += prefix + collectiona[0][0,3].join(':') + " - "
        elsif (collectiona.map{|a| a[0, 2]}.uniq.size == 1)
	  clean_allele += prefix + collectiona[0][0,2].join(':') + " - "
        elsif (collectiona.map{|a| a[0, 1]}.uniq.size == 1)
	  clean_allele += prefix + collectiona[0][0,1].join(':') + " - "
        else # darn, ambiguous
          raise 'Code problem, this shouldnt happen'
	  ambig = '1'
	  collectiona.sort!
	  collectiona.pop
	  # retry #hopefully this works.
        end
      end
      collectionb = alleles.map do |a|
        tmp = a.split('-')
        aa = tmp[0].gsub!(/[^\d:]/, '')
        tmp[1].gsub!(/[^\d:]/, '').split(':')
        if (aa =~ /^#{clean_allele[0 .. -4].gsub!(/[^\d:]/, '')}/)
          tmp[1].split(':')
        else
          nil
        end
      end
      collectionb.delete_if {|a| a == nil }
      0.upto(0) do
        if (collectionb.map{|a| a[0,4]}.uniq.size == 1)
	  clean_allele += prefix + collectionb[0][0,4].join(':')
        elsif (collectionb.map{|a| a[0,3]}.uniq.size == 1)
	  clean_allele += prefix + collectionb[0][0,3].join(':')
        elsif (collectionb.map{|a| a[0,2]}.uniq.size == 1)
	  clean_allele += prefix + collectionb[0][0,2].join(':')
        elsif (collectionb.map{|a| a[0,1]}.uniq.size == 1)
	  clean_allele += prefix + collectionb[0][0,1].join(':')
        else #darn, ambiguous
          raise 'Code problem, this shouldnt happen'
	  ambig = '1'
	  collectionb.sort!
	  collectionb.pop
	  # retry #hopefully this works.
        end
      end
      clean_allele.gsub!("\\", '')
      entry.push(clean_allele)
      entry.push(ambig)
    end
  end

  puts 'Finding Homozygousity'
  # OK, now we must find homozygousity.  IE: Cw*0722 - Cw*0722
  [[hla_a_processing, 'a'],
   [hla_b_processing, 'b'],
   [hla_c_processing, 'c']].each do |processing|
    processing[0].each do |entry|
      homozygous = '0'
      # Lets say if we detect two of the same mixtures, its heterozygous
      entry[5].each do |a|
        a[1].each do |allele|
	  tmp = allele.split(' - ')
	  if (tmp[0] == tmp[1])
	    homozygous = '1'
	  end
        end
      end
      entry.push(homozygous)
    end
  end

  # Find B*5701
  puts "Finding B*5701's"
  hla_b_processing.each do |entry|
    dist_from_b5701 = [0,0,0]
    seq = entry[2]
    [B570101, B570102, B570103].each_with_index do |cons_b5701, dex|
      0.upto(cons_b5701.size - 1) do |i|
        if (cons_b5701[i,1] == ',')
          next
        end
        if (AMBIG[cons_b5701[i, 1]].all? {|a| AMBIG[seq[i,1]].include?(a)} or
            AMBIG[seq[i, 1]].all? {|a| AMBIG[cons_b5701[i, 1]].include?(a)})
        else
          dist_from_b5701[dex] += 1
        end
      end
    end
    dist_from_b5701 = dist_from_b5701.min
    b5701 = '0'
    entry[5].each do |a|
      if options[:verbose]
        puts "   #{entry[1]}: checking for 'B*57:01' in: #{a[1]}"
      end
      a[1].each do |allele|
        if (allele.include?('B*57:01'))
	  b5701 = '1'
        end
      end
    end
    entry.push(b5701)
    entry.push(dist_from_b5701)
    puts "   closest match distance: dist: #{dist_from_b5701}, isB5701: #{b5701}" if options[:verbose]
  end

  hla_b_processing.each do |entry|
=begin
	puts "Processing #{entry[1]}";
	min = [9999, []]
	hla_consensus.each do |cons|
		if(cons[0] == entry[3]) #Look for exact matches first
			min[0] = 0
			min[1].push(cons)
		end
	end

	if(min[0] != 0) #Otherwise....
		hla_consensus.each do |cons|
			mismatches = 0
			hla_vars.each do |loc|
				break if (loc == nil)
				if(cons[0][loc, 1] != entry[3][loc, 1])
					mismatches += 1
				end
				break if(mismatches > min[0])
			end

			if(mismatches == min[0])
				min[1].push(cons)
			elsif(mismatches < min[0])
				min[0] = mismatches
				min[1] = [cons]
			end
		end
	end
	entry.push(min[0]) #mismatches_count
	entry.push(min[1]) #mismatch from X
=end
    entry.push('')
    entry.push('')
  end

  ofname = File.join(HLA_WORKFINAL_DIR, 'hla_a_seq.txt')
  puts "writing #{ofname} (#{hla_a_processing.size} entries)" if options[:verbose]
  File.open(ofname, 'w') do |file|
    hla_a_processing.each do |entry|
      alleles_all = entry[5].map {|a| a[1].join(';')}.join(';')
      alleles_all = alleles_all[0 .. 3920].gsub(/;[^;]+$/, ';...TRUNCATED') if (alleles_all.size > 3900)
      tmp = [entry[1],entry[6],alleles_all,entry[7],entry[8],entry[4][0], entry[4][1],entry[2]]
      file.puts tmp.join(',')
    end
  end

  ofname = File.join(HLA_WORKFINAL_DIR, 'hla_b_seq.txt')
  puts "writing #{ofname} (#{hla_b_processing.size} entries)" if options[:verbose]
  File.open(ofname, 'w') do |file|
    hla_b_processing.each do |entry|
      alleles_all = entry[5].map {|a| a[1].join(';')}.join(';')
      alleles_all = alleles_all[0 .. 3920].gsub(/;[^;]+$/, ';...TRUNCATED') if (alleles_all.size > 3900)
      # puts entry.inspect()
      # tmp = [entry[1],entry[6],alleles_all,entry[7],entry[8],entry[4][0],
      # entry[4][1],entry[9],entry[10], entry[2], entry[11],
      # entry[12].map {|a| a[1].join(';')}.join(';')]

      reso_status = nil
      # and HLA_ALLELES_B.alleles_clean like '%B*57%'
      # and HLA_ALLELES_B.alleles_clean not like '%B*57:01%'
      # ckw added 20-Nov-2018
      # if( !entry[6].include?('B*57:01') and entry[6].include?('B*57') )
      # ckw added 24-May-2019 at request of Chanson
      if entry[6].include?('B*57')
        reso_status = 'pending'
      end
      tmp = [entry[1],entry[6],alleles_all,entry[7],entry[8],entry[4][0],
             entry[4][1],entry[9],entry[10], entry[2], reso_status, nil]
      file.puts tmp.join(',')
    end
  end

  ofname = File.join(HLA_WORKFINAL_DIR, 'hla_c_seq.txt')
  puts "writing #{ofname} (#{hla_c_processing.size} entries)" if options[:verbose]
  File.open(ofname, 'w') do |file|
    hla_c_processing.each do |entry|
      alleles_all = entry[5].map {|a| a[1].join(';')}.join(';')
      alleles_all = alleles_all[0 .. 3920].gsub(/;[^;]+$/, ';...TRUNCATED') if (alleles_all.size > 3900)
      tmp = [entry[1],entry[6],alleles_all,entry[7],entry[8],
             entry[4][0], entry[4][1],entry[2]]
      file.puts tmp.join(',')
    end
  end

  unless options[:usedb]
    puts 'skipping db upload (--nodb)'
    puts '  also skipping the copying of files to the processed directory'
    exit(0)
  end
  puts 'Starting db upload...'

  # Upload the data
  # Hrm, password....
  # puts "Please enter your username and password in the dialog box"
  # username, password = get_login
  # NOTE: 03-08-2023 wscott this used to be an endless loop..
  max_tries = 5
  num_tries = 0
  good = false
  while !good && num_tries < max_tries
    # login_hash = Hash.new()
    # login_hash[:username] = username
    # login_hash[:password] = password
    begin
      num_tries += 1
      login(login_hash) do |dbh, hash|
        good = true
        File.open(HLA_TEMP_LOG_FILE, 'w') do |file|
          file.puts log.string
        end

        unless automode
          puts 'Database Upload (Y/N) ?'
          ans = gets.strip().upcase()
          if (ans == 'N')
            puts 'skipping upload'
            return
          else
            puts 'Continuing with database upload..'
          end
        end

        service_name = ENV['ORACLE_SERVICE']
        File.open(HLA_TEMP_LOG_FILE, 'a') do |out|
          ['a', 'b', 'c'].each do |hla_type|
            sqlfname = File.join(HLA_PROG_DIR, "hla_#{hla_type}_seq.ctl")
            logfname = File.join(HLA_LOG_DIR, "hla_#{hla_type}_seq.log")
            sql_output, sql_status = Open3.capture2e(
              SQLLDR_EXECUTABLE,
              "#{login_hash[:username]}/#{login_hash[:password]}@#{service_name}",
              "control=#{sqlfname}",
              "errors=1000",
              "log=#{logfname}")
            out.write(sql_output)
            unless sql_status.success?
              puts sql_output
              raise ("#{SQLLDR_EXECUTABLE} failed on #{sqlfname} with exit " +
                     "code #{sql_status.exitstatus}.")
            end
          end
        end

        res = system("ruby #{HLA_GEN_REPORTS_SCRIPT} --auto -t -l \"#{login_hash[:username]}/#{login_hash[:password]}\"")
        puts "#{HLA_GEN_REPORTS_SCRIPT} returned #{res}"

        # Produce a report!
        system("notepad #{HLA_TEMP_LOG_FILE}") unless automode
        FileUtils.copy(Dir[HLA_TEMP_LOG_FILE], HLA_LOG_DIR)

        date_str = Time.now.strftime('%Y-%m-%d_%H%M%S')
        puts "Moving files to processed directory with date string #{date_str}"
        [
          {
	    dir: File.join(HLA_PROCESSED_DIR, "text", date_str),
	    wc: File.join(HLA_WORKTEXT_DIR, "*")
          },
          {
	    dir: File.join(HLA_PROCESSED_DIR, "final", date_str),
	    wc: File.join(HLA_WORKFINAL_DIR, "*")
          },
        ].each{|file_kind|
          puts "   Copying #{wc}  to #{dir}" if options[:verbose]
          FileUtils.mkdir_p(file_kind[:dir])
          FileUtils.copy(Dir[file_kind[:wc]], file_kind[:dir])
          FileUtils.rm(Dir[file_kind[:wc]])
     }
      end
    rescue => e
      puts "Database upload failed: #{e}"
      puts $!.backtrace
      retry if num_tries < max_tries
    end
  end
  puts 'Upload completed..'

rescue => e
  puts "Some kind of error #{e}!"
  puts $!.to_s
  puts $!.backtrace
end

# gets if(!autodir)
