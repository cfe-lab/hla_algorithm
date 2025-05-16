=begin
Seems to work.  Needs validation and cleanup probably.  Maybe a version number?
=end

require 'digest'
require 'cfe_sequence_utils'


class HLAAlgorithm

  HLA_A_LENGTH = 787
  MIN_HLA_BC_LENGTH = 787
  MAX_HLA_BC_LENGTH = 796
  EXON2_LENGTH = 270
  EXON3_LENGTH = 276

  NUC2BIN = {
    'a' => 0b0001,
    'c' => 0b0010,
    'g' => 0b0100,
    't' => 0b1000,
    'm' => 0b0011,
    'r' => 0b0101,
    'w' => 0b1001,
    's' => 0b0110,
    'y' => 0b1010,
    'k' => 0b1100,
    'v' => 0b1110,
    'h' => 0b1101,
    'd' => 0b1011,
    'b' => 0b0111,
    'n' => 0b1111
  }
  BIN2NUC = NUC2BIN.invert

  def self.calculate_md5(io)
    md5 = Digest::MD5.new
    loop do
      chunk = io.read(1024)
      if chunk.nil?
        return md5.hexdigest
      end
      md5 << chunk
    end
  end

  def self.find_hla_std_paths(
    config_path,
    work_path,
    unzip_program
  )
    hla_std_paths = {}
    'abc'.each_char do |letter|
      zip_path = File.join(config_path, "hla_#{letter}_std_reduced.zip")
      csv_path = File.join(config_path, "hla_#{letter}_std_reduced.csv")
      hla_std_paths[letter] = csv_path  # default
      unless File.exists?(zip_path)
        next
      end
      md5_zip = nil
      Open3.popen2(unzip_program, '-p', zip_path) do |stdin, stdout|
        md5_zip = calculate_md5(stdout)
      end
      md5_csv = nil
      if File.exists?(csv_path)
        File.open(csv_path) do |f|
          md5_csv = calculate_md5(f)
        end
      end
      if md5_zip == md5_csv
        next
      end
      unless system(unzip_program, '-qqo', zip_path, '-d', work_path)
        raise "Unable to unzip HLA reference #{zip_path}."
      end
      csv_path = File.join(work_path, "hla_#{letter}_std_reduced.csv")
      hla_std_paths[letter] = csv_path
    end
    return hla_std_paths
  end

  # Check if a sequence is the right length.
  def self.check_length(letter, seq, name)
    err = false
    if name.downcase.end_with?("short")
      if letter == "A" 
        err = (seq.length >= HLA_A_LENGTH)
      elsif name.include? "exon2"
        err = (seq.length >= EXON2_LENGTH)
      elsif name.include? "exon3" 
        err = (seq.length >= EXON3_LENGTH)
      else 
        err = (seq.length >= MAX_HLA_BC_LENGTH)
      end

    elsif letter == "A"
      err = (seq.length != HLA_A_LENGTH)
    elsif name.include? "exon2"
      err = (seq.length != EXON2_LENGTH)
    elsif name.include? "exon3"
      err = (seq.length != EXON3_LENGTH)
    else
      err = ((seq.length > MAX_HLA_BC_LENGTH) or 
            (seq.length < MIN_HLA_BC_LENGTH))
    end

    if err
      err = "Sequence %s is the wrong length (%d bp). Check the locus %s." % [name, seq.length, letter]
      # $log.puts err
      puts err
      return false
    end
    return true
  end
  
  def nuc2bin(nuc_seq)
    bin_seq = []
    0.upto(nuc_seq.size - 1) do |i|
      bin_seq << NUC2BIN[nuc_seq[i,1]]
    end
    return bin_seq
  end

  def bin2nuc(bin_seq)
    nseq = ""
    0.upto(bin_seq.length - 1) do |i|
      nseq = nseq + BIN2NUC[bin_seq[i]]
    end
    return nseq
  end

  # Find all standards which have less than 5 mismatches to the query sequence.
  def get_matching_stds(seq, hla_stds)
    matching_stds = []
    hla_stds.each do |std|
      allele, std_seq = std
      mismatches = std_match(std_seq, seq)
      if mismatches < 5
        matching_stds.push([allele, std_seq, mismatches])
      end
    end
    return matching_stds.sort_by{|s| s[2]}
  end


  def initialize(
    hla_a_std_path=nil,
    hla_b_std_path=nil,
    hla_c_std_path=nil,
    hla_freq_path=nil
  )
    @hla_stds = {
      a: [],
      b: [],
      c: []
    }

    # hla_std_paths = find_hla_std_paths()
    hla_std_paths = {
      a: hla_a_std_path,
      b: hla_b_std_path,
      c: hla_c_std_path
    }
    if hla_a_std_path.nil?
      hla_std_paths[:a] = 'hla_algorithm/hla_a_std_reduced.csv'
    end
    if hla_b_std_path.nil?
      hla_std_paths[:b] = 'hla_algorithm/hla_b_std_reduced.csv'
    end
    if hla_c_std_path.nil?
      hla_std_paths[:c] = 'hla_algorithm/hla_c_std_reduced.csv'
    end

    [:a, :b, :c].each do |gene|
      File.open(hla_std_paths[gene]) do |std_file|
        std_file.each_line do |line|
          tmp = line.strip.split(',')
          @hla_stds[gene].push([tmp[0], tmp[1] + tmp[2]])
        end
      end
    end

    @hla_freqs = [{},{},{}]
    freq_file = hla_freq_path
    if hla_freq_path.nil?
      freq_file = 'hla_algorithm/hla_frequencies.csv'
    end
    File.open(freq_file) do |file|
      file.each_line do |line|
        row = line.split(',').map{|a| a.strip }
        0.upto(2) do |column|
          tmp = row[column*2, 2].map {|a| [a[0,2], a[2,2]] }
          @hla_freqs[column][tmp] = 0 if @hla_freqs[column][tmp] == nil
          @hla_freqs[column][tmp] += 1
        end
      end
    end

    @b570101 = 'GCTCCCACTCCATGAGGTATTTCTACACCGCCATGTCCCGGCCCGGCCGCGGGGAGCCCCGCTTCATCGCAGTGGGCTACGTGGACGACACCCAGTTCGTGAGGTTCGACAGCGACGCCGCGAGTCCGAGGATGGCGCCCCGGGCGCCATGGATAGAGCAGGAGGGGCCGGAGTATTGGGACGGGGAGACACGGAACATGAAGGCCTCCGCGCAGACTTACCGAGAGAACCTGCGGATCGCGCTCCGCTACTACAACCAGAGCGAGGCCG,GGTCTCACATCATCCAGGTGATGTATGGCTGCGACGTGGGGCCGGACGGGCGCCTCCTCCGCGGGCATGACCAGTCCGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGAGCTCCTGGACCGCGGCGGACACGGCGGCTCAGATCACCCAGCGCAAGTGGGAGGCGGCCCGTGTGGCGGAGCAGCTGAGAGCCTACCTGGAGGGCCTGTGCGTGGAGTGGCTCCGCAGATACCTGGAGAACGGGAAGGAGACGCTGCAGCGCGCGG'
    @b570102 = 'GCTCCCACTCCATGAGGTATTTCTACACCGCCATGTCCCGGCCCGGCCGCGGGGAGCCCCGCTTCATCGCAGTGGGCTACGTGGACGACACCCAGTTCGTGAGGTTCGACAGCGACGCCGCGAGTCCGAGGATGGCGCCCCGGGCGCCATGGATAGAGCAGGAGGGGCCGGAGTATTGGGACGGGGAGACACGGAACATGAAGGCCTCCGCGCAGACTTACCGAGAGAACCTGCGGATCGCGCTCCGCTACTACAACCAGAGCGAGGCCG,GGTCTCACATCATCCAGGTGATGTATGGCTGCGACGTGGGGCCGGACGGGCGCCTCCTCCGCGGGCATGACCAGTCTGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGAGCTCCTGGACCGCGGCGGACACGGCGGCTCAGATCACCCAGCGCAAGTGGGAGGCGGCCCGTGTGGCGGAGCAGCTGAGAGCCTACCTGGAGGGCCTGTGCGTGGAGTGGCTCCGCAGATACCTGGAGAACGGGAAGGAGACGCTGCAGCGCGCGG'
    @b570103 = 'GCTCCCACTCCATGAGGTATTTCTACACCGCCATGTCCCGGCCCGGCCGCGGGGAGCCCCGCTTCATCGCAGTGGGCTACGTGGACGACACCCAGTTCGTGAGGTTCGACAGCGACGCCGCGAGTCCGAGGATGGCGCCCCGGGCGCCATGGATAGAGCAGGAGGGGCCGGAGTATTGGGACGGGGAGACACGGAACATGAAGGCCTCCGCGCAGACTTACCGAGAGAACCTGCGGATCGCGCTCCGCTACTACAACCAGAGCGAGGCCG,GGTCTCACATCATCCAGGTGATGTATGGCTGCGACGTGGGGCCGGACGGGCGCCTCCTCCGCGGGCATGACCAGTCCGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGAGCTCCTGGACCGCGGCGGACACGGCGGCTCAGATCACCCAGCGCAAGTGGGAGGCGGCCCGTGTGGCGGAGCAGCTGAGAGCCTACCTGGAGGGCCTGTGTGTGGAGTGGCTCCGCAGATACCTGGAGAACGGGAAGGAGACGCTGCAGCGCGCGG'

  end

  def analyze(seqs, type='B')
    result = HLAResult.new()
    result.type = type
    result.seqs = seqs

    #Quality checking
    if(type == 'A')
      result.errors << 'Wrong number of sequences (needs 1)' if(seqs.size() != 1)
      result.errors << 'Sequence is the wrong size (should be 787)' if(seqs.size() == 1 and seqs[0].size() != 787)
    elsif(type == 'B')
      result.errors << 'Wrong number of sequences (needs 2)' if(seqs.size() != 2)
      result.errors << 'Sequence 1 is the wrong size (should be 270)' if(seqs.size() == 2 and seqs[0].size() != 270)
      result.errors << 'Sequence 2 is the wrong size (should be 276)' if(seqs.size() == 2 and seqs[1].size() != 276)
    elsif(type == 'C')
      result.errors << 'Wrong number of sequences (needs 2)' if(seqs.size() != 2)
      result.errors << 'Sequence 1 is the wrong size (should be 270)' if(seqs.size() == 2 and seqs[0].size() != 270)
      result.errors << 'Sequence 2 is the wrong size (should be 276)' if(seqs.size() == 2 and seqs[1].size() != 276)
    end
    result.errors << 'Sequence has invalid characters' if(!(seqs.join('') =~ /^[atgcrykmswnbdhv]+$/i))

    return result if(result.errors.size() > 0) #quick exit left

    #analyze.
    hla_consensus = []
    hla_stds = @hla_stds[type]
    min = [9999, []]
    entry_seq_reduced = seqs.join('') #rename to something less stupid, like "seq".
    if(type == 'A') #We don't need the middle bits.
      entry_seq_reduced = entry_seq_reduced[0 .. 270] + entry_seq_reduced[512 .. 787]
    end
    matching_stds = []
    alleles_hash = Hash.new

    hla_stds.each do |std|
      mismatch_cnt = CfeSequenceUtils::std_match(std[1], entry_seq_reduced)
#      puts "#{std[0]} - #{mismatch_cnt}"
#      puts std[1]
#      puts entry_seq_reduced
#      exit
      if(mismatch_cnt < 5)
        matching_stds.push([std, mismatch_cnt])
      end
    end

    #Main work + rosemary's optimizations.
    matching_stds.each_with_index do |std_ax, i_a|
      std_a, std_a_min = std_ax[0], std_ax[1]
      next if(std_a_min > min[0])
      matching_stds.each_with_index do |std_bx, i_b|
        next if(i_b < i_a) # I think this makes sense...
        std_b, std_b_min = std_bx[0], std_bx[1]
        next if(std_b_min > min[0])  #The key to Rosemary's optimizations
        std = ''
        #SUPER_COMBO
        0.upto((std_a[1].size() / 3) - 1) do |i| #This bit really takes a while.  How to improve...
          std += CfeSequenceUtils::SUPER_COMBO[[std_a[1][i * 3,3], std_b[1][i * 3,3]]]
        end
        alleles_hash[std] = [] if(alleles_hash[std] == nil)
        alleles_hash[std].push("#{std_a[0]} - #{std_b[0]}")

        #check to see if I need to add it to min
        if(std == entry_seq_reduced) #Look for exact matches first
          if(min[0] != 0)
            min[0] = 0
            min[1] = [[std, alleles_hash[std]]]
          elsif(!min[1].include?([std, alleles_hash[std]]))
            min[1].push([std, alleles_hash[std]]) #The element should get continuously updated, right?
          end
          next
        end

        mismatches = 0
        0.upto(std.size() - 1) do |dex|
          if(std[dex, 1] != entry_seq_reduced[dex, 1])
            mismatches += 1
          end
          break if(mismatches > min[0])
        end

        if(mismatches == min[0] and !min[1].include?([std, alleles_hash[std]]))
          min[1].push([std, alleles_hash[std]])
        elsif(mismatches < min[0])
          min[0] = mismatches
          min[1] = [[std, alleles_hash[std]]]
        end
      end
    end

    #okay, keep going!
    mislist = []
    min[1].each do |cons|  #really, we should just pick one, right?
      0.upto(cons[0].size - 1) do |dex|
        if(cons[0][dex, 1] != entry_seq_reduced[dex, 1])
          mislist.push("#{((dex + ((type == 'A' and dex > 270) ? 241 : 0)) + 1).to_s}:#{entry_seq_reduced[dex, 1]}")
        end
      end
      break
    end
    mislist.uniq!

    result.mismatches = mislist
    result.alleles_all = min[1].map{|a| a[1]}.flatten()

    #Continue, we must do ADDITIONAL ANALYISISIS
    #CLEAN
    prefix = type + '\*'
    fcnt = 0 #used to offset the freq column
    fcnt = 2 if(type == 'B')
    fcnt = 4 if(type == 'C')
    clean_allele = ''
    alleles = []
    ambig = false


		alleles = Array.new(result.alleles_all)

    if(alleles == [])
      result.errors << 'Could not find any matching alleles'
      return result
    end

    collection = alleles.map do |a|
      tmp = a.split('-')
      [tmp[0].gsub(/[^\d:]/, '').split(':'), tmp[1].gsub(/[^\d:]/, '').split(':')]
    end

    if(collection.map{|e| [e[0][0], e[1][0]]}.uniq.size != 1)
      ambig = true
      collection_ambig = collection.map{|e| [e[0][0 .. 1], e[1][0 .. 1]]}.uniq

      #Rosemary's code
      collection_ambig.each do |a|
        freq = @hla_freqs[fcnt/2][a]
        freq = 0 if(freq == nil)
        a.push(freq)
      end

      max_allele = collection_ambig.max do |a,b|
        if(a[2] != b[2]) #Go by frequency
          a[2] <=> b[2]
        elsif(b[0][0].to_i != a[0][0].to_i) #Then lowest first allele
          b[0][0].to_i <=> a[0][0].to_i
        elsif(b[0][1].to_i != a[0][1].to_i)
          b[0][1].to_i <=> a[0][1].to_i
        elsif(b[1][0].to_i != a[1][0].to_i) #Then lowest second allele
          b[1][0].to_i <=> a[1][0].to_i
        else
          b[1][1].to_i <=> a[1][1].to_i
        end
      end

      a1 = max_allele[0][0]
      a2 = max_allele[1][0]

      alleles.delete_if {|a| !(a =~ /^#{prefix}#{a1}:([^\s])+\s-\s#{prefix}#{a2}:/)}
    end

    #OK, make sure all the alleles mostly match
    collectiona = alleles.map do |a|
      tmp = a.split('-')
      tmp[0].gsub!(/[^\d:]/, '')
      tmp[0].split(':')
    end

    0.upto(0) do
			if(collectiona.map{|a| a[0, 4]}.uniq.size == 1)
				clean_allele += prefix + collectiona[0][0,4].join(':') + " - "
			elsif(collectiona.map{|a| a[0, 3]}.uniq.size == 1)
				clean_allele += prefix + collectiona[0][0,3].join(':') + " - "
			elsif(collectiona.map{|a| a[0, 2]}.uniq.size == 1)
				clean_allele += prefix + collectiona[0][0,2].join(':') + " - "
			elsif(collectiona.map{|a| a[0, 1]}.uniq.size == 1)
				clean_allele += prefix + collectiona[0][0,1].join(':') + " - "
			else #darn, ambiguous
        puts "wut?"
        raise "Code problem, this shouldn't happen" #uh, wut?
				ambig = true
				collectiona.sort!
				collectiona.pop
				redo #hopefully this works.
			end
		end

    collectionb = alleles.map do |a|
      tmp = a.split('-')
      aa = tmp[0].gsub!(/[^\d:]/, '')
      tmp[1].gsub!(/[^\d:]/, '').split(':')
      if(aa =~ /^#{clean_allele[0 .. -4].gsub!(/[^\d:]/, '')}/)
        tmp[1].split(':')
      else
        nil
      end
		end
    collectionb.delete_if {|a| a == nil }

    0.upto(0) do
			if(collectionb.map{|a| a[0,4]}.uniq.size == 1)
				clean_allele += prefix + collectionb[0][0,4].join(':')
			elsif(collectionb.map{|a| a[0,3]}.uniq.size == 1)
				clean_allele += prefix + collectionb[0][0,3].join(':')
			elsif(collectionb.map{|a| a[0,2]}.uniq.size == 1)
				clean_allele += prefix + collectionb[0][0,2].join(':')
			elsif(collectionb.map{|a| a[0,1]}.uniq.size == 1)
				clean_allele += prefix + collectionb[0][0,1].join(':')
			else #darn, ambiguous
        puts "wut2"
        raise "Code problem, this shouldn't happen"
				ambig = true
				collectionb.sort!
				collectionb.pop
				redo #hopefully this works.
			end
		end

    clean_allele.gsub!("\\", '')

    result.alleles_clean = clean_allele
    result.ambiguous = ambig

    #Finding homozygosity
    alleles = Array.new(result.alleles_all)
    alleles.each do |a|
      tmp = a.split(' - ')
      result.homozygous = true if(tmp[0] == tmp[1])
		end

    if(type == 'B') #Find B5701
      dist_from_b5701 = [0,0,0]
      seq_comma = seqs.join(',')
      [@b570101, @b570102, @b570103].each_with_index do |cons_b5701, dex|
        0.upto(cons_b5701.size - 1) do |i|
          next if(cons_b5701[i,1] == ',')
          if(CfeSequenceUtils::AMBIG[cons_b5701[i, 1]].all? {|a| CfeSequenceUtils::AMBIG[seq_comma[i,1]].include?(a)} 
             or CfeSequenceUtils::AMBIG[seq_comma[i, 1]].all? {|a| CfeSequenceUtils::AMBIG[cons_b5701[i, 1]].include?(a)})
          else
            dist_from_b5701[dex] += 1
          end
        end
      end

      dist_from_b5701 = dist_from_b5701.min


      alleles.each do |a|
        if(a.include?('B*57:01'))
          result.b5701 = true
        end
      end
      result.dist_b5701 = dist_from_b5701
    end


    return result
  end

end

class HLAResult
  attr_accessor :seqs, :alleles_all, :alleles_clean, :mismatches, :ambiguous, :homozygous, :type, :alg_version, :b5701, :dist_b5701, :errors
  def initialize()
    @errors = []
    @alleles_all = []
    @mismatches = []
    @b5701 = false
    @ambiguous = false
    @homozygous = false
  end
end
