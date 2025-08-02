#!/usr/bin/ruby
#
# This program downloads the newest version of hla_nuc.fasta and then
# processes it, creating three standard files for a, b, and c.  Any sequence
# that doesn't seem to align properly will be rejected.
# The alignment algorithm I'm using is the dirtiest possible.
# This program requires the bioruby library


#require 'bio'
require 'fileutils'
require 'net/http'
require 'uri'
require 'date'
require 'digest'
require 'json'

#scores the sequence according to how many characters don't match.  An
#alignment of 0 means a perfect match.  We optimize it by assuming anything under 20 is probably
#a match.
def score(seq, align)
	maxscore = align.size
	maxseq = -1;

	0.upto(seq.size - align.size) do |i|
		score = align.size
		0.upto(align.size - 1) do |j|
			if(align[j] == seq[i + j])
				score -= 1
			end
		end

		if(score < maxscore)
			maxscore = score
			maxseq = i
			if(maxscore < 20)
				return [maxscore, seq[maxseq, align.size]]
			end
		end
	end
    if(maxscore > 30)
#        puts maxscore
#        puts align
#        puts seq[maxseq, align.size]
    end
	return [maxscore, seq[maxseq, align.size]]
end

#Exon sequences to compare against(used for scoring)
a_exon2_align='GCTCCCACTCCATGAGGTATTTCTTCACATCCGTGTCCCGGCCCGGCCGCGGGGAGCCCCGCTTCATCGCCGTGGGCTACGTGGACGACACGCAGTTCGTGCGGTTCGACAGCGACGCCGCGAGCCAGAAGATGGAGCCGCGGGCGCCGTGGATAGAGCAGGAGGGGCCGGAGTATTGGGACCAGGAGACACGGAATATGAAGGCCCACTCACAGACTGACCGAGCGAACCTGGGGACCCTGCGCGGCTACTACAACCAGAGCGAGGACG'
a_exon3_align='GTTCTCACACCATCCAGATAATGTATGGCTGCGACGTGGGGCCGGACGGGCGCTTCCTCCGCGGGTACCGGCAGGACGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGCGCTCTTGGACCGCGGCGGACATGGCAGCTCAGATCACCAAGCGCAAGTGGGAGGCGGTCCATGCGGCGGAGCAGCGGAGAGTCTACCTGGAGGGCCGGTGCGTGGACGGGCTCCGCAGATACCTGGAGAACGGGAAGGAGACGCTGCAGCGCACGG'
b_exon2_align='GCTCCCACTCCATGAGGTATTTCTACACCTCCGTGTCCCGGCCCGGCCGCGGGGAGCCCCGCTTCATCTCAGTGGGCTACGTGGACGACACCCAGTTCGTGAGGTTCGACAGCGACGCCGCGAGTCCGAGAGAGGAGCCGCGGGCGCCGTGGATAGAGCAGGAGGGGCCGGAGTATTGGGACCGGAACACACAGATCTACAAGGCCCAGGCACAGACTGACCGAGAGAGCCTGCGGAACCTGCGCGGCTACTACAACCAGAGCGAGGCCG'
b_exon3_align='GGTCTCACACCCTCCAGAGCATGTACGGCTGCGACGTGGGGCCGGACGGGCGCCTCCTCCGCGGGCATGACCAGTACGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGCGCTCCTGGACCGCCGCGGACACGGCGGCTCAGATCACCCAGCGCAAGTGGGAGGCGGCCCGTGAGGCGGAGCAGCGGAGAGCCTACCTGGAGGGCGAGTGCGTGGAGTGGCTCCGCAGATACCTGGAGAACGGGAAGGACAAGCTGGAGCGCGCTG'
c_exon2_align='GCTCCCACTCCATGAAGTATTTCTTCACATCCGTGTCCCGGCCTGGCCGCGGAGAGCCCCGCTTCATCTCAGTGGGCTACGTGGACGACACGCAGTTCGTGCGGTTCGACAGCGACGCCGCGAGTCCGAGAGGGGAGCCGCGGGCGCCGTGGGTGGAGCAGGAGGGGCCGGAGTATTGGGACCGGGAGACACAGAAGTACAAGCGCCAGGCACAGACTGACCGAGTGAGCCTGCGGAACCTGCGCGGCTACTACAACCAGAGCGAGGCCG'
c_exon3_align='GGTCTCACACCCTCCAGTGGATGTGTGGCTGCGACCTGGGGCCCGACGGGCGCCTCCTCCGCGGGTATGACCAGTACGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGCGCTCCTGGACCGCCGCGGACACCGCGGCTCAGATCACCCAGCGCAAGTGGGAGGCGGCCCGTGAGGCGGAGCAGCGGAGAGCCTACCTGGAGGGCACGTGCGTGGAGTGGCTCCGCAGATACCTGGAGAACGGGAAGGAGACGCTGCAGCGCGCGG'

Dir.chdir "config"
filename = 'hla_nuc.fasta'
timestamp_path = filename + '.mtime'

repo_path = "https://raw.githubusercontent.com/ANHIG/IMGTHLA"

# Find latest release at https://github.com/ANHIG/IMGTHLA/releases
hla_nuc_version = File.read('hla_nuc.fasta.version.txt').strip
puts "attempting to download version #{hla_nuc_version} from #{repo_path}"

uri = URI.parse("#{repo_path}/#{hla_nuc_version}/hla_nuc.fasta")
response = Net::HTTP.get_response(uri)
md5 = Digest::MD5.new
Net::HTTP.get_response(uri) do |response|
	response.value  # Raise error if not 200 response code.
	File.open(filename, 'w') do |file|
		response.read_body do |segment|
			file.write(segment)
			md5 << segment
		end
	end
end
checksum_report = md5.hexdigest + '  ' + filename + "\n"
File.write('hla_nuc.fasta.checksum.txt', checksum_report)
puts "Parsing " + filename;

hla_a = []
hla_b = []
hla_c = []

diff_reject = 32


#file = Bio::FastaFormat.open(filename)
fasta = []
enu=[]
File.open(filename) do |file|
  file.each_line do |line|
    if(line =~ /^>/)
      fasta.push(enu)
      enu = [line.strip, '']
    else
      enu[1] += line.strip
    end
  end
  fasta.push(enu)
end

fasta.delete_if{|e| e== []}

bar_width = 50
fasta.each_with_index do |entry, index| #for each fasta sequence
	#title = entry.definition[entry.definition.index(' ') + 1 .. entry.definition.size]
    title = entry[0].split(' ')[1]
	type = title[0, 1]
	data = entry[1]
	progress_bar = ('#' * (index.to_f/fasta.length*bar_width) + '.' * bar_width)
	progress_bar = progress_bar[0...bar_width]

	data.tr!("\n\t\r ", '') #get rid of whitespace
	message = ''

	if(type == 'A')
		exon2 = score(data, a_exon2_align)
		exon3 = score(data, a_exon3_align)
		if(exon2[0] <= diff_reject and exon3[0] <= diff_reject)
            message = "Approving " + title + ":  " + exon2[0].to_s + " " + exon3[0].to_s
			hla_a.push( [title, exon2[1], exon3[1]] )
		else
			message = "***Rejecting " + title + ":  " + exon2[0].to_s + " " + exon3[0].to_s
		end
	elsif(type == 'B')
		exon2 = score(data, b_exon2_align)
		exon3 = score(data, b_exon3_align)
		if(exon2[0] <= diff_reject and exon3[0] <= diff_reject)
            message = "Approving " + title + ":  " + exon2[0].to_s + " " + exon3[0].to_s
			hla_b.push( [title, exon2[1], exon3[1]] )
		else
			message = "***Rejecting " + title + ":  " + exon2[0].to_s + " " + exon3[0].to_s
		end
	elsif(type == 'C')
		exon2 = score(data, c_exon2_align)
		exon3 = score(data, c_exon3_align)
		if(exon2[0] <= diff_reject and exon3[0] <= diff_reject)
            message = "Approving " + title + ":  " + exon2[0].to_s + " " + exon3[0].to_s
			hla_c.push( [title, exon2[1], exon3[1]] )
		else
			message = "***Rejecting " + title + ":  " + exon2[0].to_s + " " + exon3[0].to_s
		end
	end
	print "\r" + progress_bar + ' ' + message + ".     "
end
puts "\r" + '#' * bar_width + ' Completed.' + ' ' * 18

#file.close

#Lets sort, just to make things easier for our eyes
hla_a.sort!
hla_b.sort!
hla_c.sort!

File.open('hla_a_std.csv', 'w') do |file|
	hla_a.each do |entry|
		file.puts entry[0] + "," + entry[1] + "," + entry[2]
	end
end

File.open('hla_b_std.csv', 'w') do |file|
	hla_b.each do |entry|
		file.puts entry[0] + "," + entry[1] + "," + entry[2]
	end
end

File.open('hla_c_std.csv', 'w') do |file|
	hla_c.each do |entry|
		file.puts entry[0] + "," + entry[1] + "," + entry[2]
	end
end

File.delete(filename)
