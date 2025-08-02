=begin
This file will generate report pdfs for HLA

2023-09-01 wscott changes for running in docker and Hermes
2023-09-22 wscott changes for Hermes: removed index_find_enum calls
=end

require 'fileutils'
require 'logger'
require 'optparse'

require 'ckwlib/login'
require 'ckwlib/report_encrypt_manager'
require 'ckwlib/cfe_env'

LAB_UPLOAD_ROOT = $cfe_env[:lab_upload_root]
raise 'LAB_UPLOAD_ROOT is not defined' if LAB_UPLOAD_ROOT.nil?
HLA_PROG_DIR = File.join(LAB_UPLOAD_ROOT, 'hla')

HLA_LOG_DIR = $cfe_env[:output_hla_log_dir]
raise 'HLA_LOG_DIR is not defined' if HLA_LOG_DIR.nil?

logger = Logger.new(File.join(HLA_LOG_DIR, 'generate_hla_reports.log'), 'daily')

#
REPORT_CODE_DIR = $cfe_env[:reports_root]
raise 'REPORT_CODE_DIR is not defined' if REPORT_CODE_DIR.nil?

REPORT_GEN_PROGRAM = File.join(REPORT_CODE_DIR, 'cfe_hla_report.rb')
raise "Error: #{REPORT_GEN_PROGRAM} not found!"\
  unless File.exist?(REPORT_GEN_PROGRAM)

# FileUtils.cd(BASE_PATH)

# command line options
options = {}
OptionParser.new do |opts|
  opts.banner = 'Usage: generate_hla_reports.rb [options]'
  opts.on("-l", "--login OPT", "Login (user/pass)") do |v|
    options[:login] = v
  end
  opts.on("-e", "--enum OPT", "Enum") do |v|
    options[:enum] = v
  end
  opts.on("-d", "--date OPT", 'Upload Date (DD-MON-YYYY)') do |v|
    options[:date] = v
  end
  opts.on("-t", "--today", "Generate reports uploaded today") do |v|
    options[:today] = true
  end
  opts.on("-r", "--replace", "Replace current reports") do |v|
    options[:replace] = true
  end
  opts.on("-a", "--auto", "Automate, no user input") do |v|
    options[:auto] = true
  end
end.parse!

# check some input arguments
if options[:enum]
  if options[:date] || options[:today]
    puts 'Error: cannot provide both enum and date (-d or -t) options'
    exit(1)
  end
end

if options[:date]
  if options[:today]
    puts 'Error: cannot provide both -d and -t options'
    exit(1)
  end
  raise 'invalid upload date (DD-MON-YYYY required)'\
    unless $cfe_env.check_dd_mon_yyyy_string(options[:date])
end

# Set up the DB login if given on command line.
login_hash = {}
if (options[:login])
  tmp = options[:login].split('/')
  login_hash[:username] = tmp[0]
  login_hash[:password] = tmp[1]
end

login(login_hash) do |dbh|
  # Figure out what enums we are doing.
  opt = ''
  if (options[:today])
    opt = 'A'
  elsif (options[:date])
    opt = 'A'
  elsif (options[:enum])
    opt = 'B'
  else
    puts 'A)  Generate a set of reports by date'
    puts 'B)  Generate a single report by enumber'
    opt = STDIN.gets().strip().upcase()
  end

  enums = []
  query_param_enum = nil
  query_param_date = nil
  if (opt == 'A')
    if (options[:today])
      query_param_date = Date.today().strftime('%d-%b-%Y').upcase
    elsif (options[:date])
      query_param_date = options[:date]
    else
      puts 'Enter the upload date (DD-MMM-YYYY):'
      query_param_date = STDIN.gets().strip().upcase()
    end
  elsif (opt == 'B')
    if (options[:enum])
      query_param_enum = options[:enum]
    else
      puts 'Enter the enumber:'
      query_param_enum = STDIN.gets().strip().upcase()
    end
  else
    puts 'Okay, quitting.'
    exit(0)
  end

  if query_param_date
    # unless query_param_date !~ /^[0-9]{2}-[A-Z]{3}-[0-9]{4}$/ then
    unless $cfe_env.check_dd_mon_yyyy_string(query_param_date) then
      puts 'Date format incorrect, please use DD-MON-YYYY, quitting.'
      exit(1)
    end
  end
 
  # Currently, only one of enum or data params can be supplied, so we don't
  # need to combine them.
  db_filter = ''
  db_params = []
  if ! query_param_enum.nil?
    db_params << query_param_enum
    db_filter = 'and hab.enum = ?'
  elsif ! query_param_date.nil?
    db_params << query_param_date
    db_filter = "and trunc(hab.enterdate) = TO_DATE(?, 'DD-MON-YYYY')"
  end
  sql = <<-SQL
      select distinct
        hab.enum              enum,
        lpi.do_not_report     do_not_report
      from
        lab_progress lp
        JOIN specimen.hla_alleles_b hab ON lp.enum = hab.enum
        JOIN specimen.lab_progress_info lpi ON lp.enum = lpi.enum
      where
        lp.test_code='HLA'
        and lp.status in ('1','2','3','4','5','11')
        #{db_filter}
      SQL

  sth = dbh.execute(sql, *db_params)
  while (row = sth.fetch())
    enums << row[0] if(row[1] != 'Y')
  end

  if enums.empty?
    puts 'Could not find any enumbers, quitting.'
    exit(0)
  end

  # em = EncryptManager.new(
  #  File.join(REPORT_ENCRYPT_MANAGER_STORE_PATH, 'index.hla.dat'),
  #  REPORT_ENCRYPT_MANAGER_STORE_PATH,
  #  dbh
  # )
  enc_hash = get_encrypt_manager_hash(dbh, apc=nil, be_verbose=false)
  em = enc_hash['HLA']
  raise 'Error: failed to determine EncryptionManager for HLA' if em.nil?

  #Okay, now for the fun stuff.
  enums.each do |enum|
    # Check to see if report already exists, if so, warn user and ask if we should skip.
    if (em.has_report(enum) && !options[:replace])
      unless options[:auto]
        puts 'Report for #{enum} already exists, do you want to:'
        puts 'A) REPLACE the old report with the new one.'
        puts 'B) SKIP generation of the new one.'
        opt2 = STDIN.gets().strip().upcase()
        if (opt2 == 'A')
          # keep on going
        elsif (opt2 == 'B')
          next
        else
          puts 'Invalid selection, quitting'
          STDIN.gets()
          exit(0)
        end
      else
        puts "#{enum} could not be written, as report already exists."
        next # skip
      end
    end

    retries = 2
    while (retries >= 0)
      # retry
      begin
        # generate pdf in the work directory.
        # FileUtils.cd(BASE_PATH)

        puts "generating report for #{enum}..."
        workdir = $cfe_env.gen_secure_work_path()
        res = system("ruby #{REPORT_GEN_PROGRAM} -e #{enum} -s #{workdir} --login \"#{login_hash[:username]}/#{login_hash[:password]}\"")
        if res.nil? || !res
          puts "report generation failed with #{res}"
        end
        pdf_file = File.join(workdir, "#{enum}.pdf")
        raise "Error producing report for #{enum}: #{pdf_file} not found"\
          unless File.exist?(pdf_file)
        hermes_response = em.encrypt_pdf(pdf_file, enum)
        fileid = hermes_response[:data]['file_id']

        # error checking
        # 2023-09-22 wscott no longer needed.
        #tmp_file = em.index_find_enum(enum)
        #if (!File.exist?(tmp_file))
        #  raise "#{enum}: Could not find encrypted file!!!\n"
        #elsif (File.size(tmp_file) < 1000 )
        #  # at least 1000 bytes
        #  raise "#{enum}: Encrypted file appears to be corrupted!!!\n"
        #end

        retries = -100 # all good
        puts "Generated report on enum #{enum}, fileid: #{fileid}."
      rescue => e
        puts "Error encountered: #{e}"
        puts $!
        puts $!.backtrace
        logger.error "#{$!} (#{e}: retries remaining #{retries})"
        retries -= 1
      end
    end
    # end retries

    stmt01 = <<-SQL
      delete from
        SPECIMEN.LAB_REPORT_INFO
      where enum = :enum
      SQL
    sth = dbh.execute(stmt01, enum)
    sth.finish()

    stmt02 = <<-SQL
      insert into SPECIMEN.LAB_REPORT_INFO
      (
        ENUM,
        GEN_DATE,
        APPROVE_DATE,
        ENTERDATE,
        ENTERED_BY
      ) values (
        :enum,
        sysdate,
        null,
        sysdate,
        'script'
      )
      SQL
    sth = dbh.execute(stmt02, enum)
    sth.finish()
    dbh.commit()
  end
end
