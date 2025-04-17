Dir.chdir "config"
['a','b','c'].each do |letter|
    data = []
    source_filename = "hla_#{letter}_std.csv"
    puts "reading #{source_filename}..."
    File.open(source_filename) do |file|
        file.each_line do |line|
            data << line.strip.split(',')
        end
    end

    data.sort! do |a,b|
        tmpa = a[0].split(':')
        tmpb = b[0].split(':')
        (tmpa[0] != tmpb[0] ? tmpa[0].split('*')[1].to_i <=> tmpb[0].split('*')[1].to_i :
            (tmpa[1] != tmpb[1] ? tmpa[1].to_i <=> tmpb[1].to_i :
            (tmpa[2] != tmpb[2] ? tmpa[2].to_i <=> tmpb[2].to_i :
            tmpa[3].to_i <=> tmpb[3].to_i )))
    end

    #File.open("hla_std_#{letter}_test.csv", 'w') do |file|
        #data.each do |d|
            #file.puts d.join(',')
        #end
    #end

    data.each_with_index do |orig, i|
        next if(orig == nil)
        orig_allele = orig[0]
        orig_seq = orig[1 .. 2]
        match_count = 0
        data.each_with_index do |row, j|
            next if(j <= i or row == nil)
            if(orig_seq == row[1 .. 2])
                match_count += 1
                puts "Reducing #{row[0]} into #{orig_allele}"
                data[j] = nil
            end
        end
        if(match_count > 0)
            gallele = orig_allele.split(':')
            gallele = gallele[0 .. 2] if(gallele.size > 3)
            gallele = gallele.join(':') + 'G'
            puts "Turning #{orig_allele} into #{gallele}"
            orig[0] = gallele
        end

    end

    data.delete(nil)

    reduced_filename = "hla_#{letter}_std_reduced.csv"
    puts "writing #{reduced_filename}"
    File.open(reduced_filename, 'w') do |file|
        data.each do |d|
            file.puts d.join(',')
        end
    end

    File.delete(source_filename)
    system('zip', "hla_#{letter}_std_reduced.zip", reduced_filename)
end
