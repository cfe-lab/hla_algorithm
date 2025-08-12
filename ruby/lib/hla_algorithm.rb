require 'digest'
require 'open3'
require 'json'


HLA_INTERPRET_FROM_JSON = ENV['HLA_INTERPRET_FROM_JSON']
if HLA_INTERPRET_FROM_JSON.nil?
  raise "HLA_INTERPRET_FROM_JSON must be set"
end


class HLAResult
  attr_accessor(
    :seqs,
    :alleles_all,
    :alleles_clean,
    :alleles_for_mismatches,
    :mismatches,
    :ambiguous,
    :homozygous,
    :locus,
    :alg_version,
    :b5701,
    :dist_b5701,
    :errors
  )

  def initialize(raw_result)
    @seqs = raw_result["seqs"]
    @alleles_all = raw_result["alleles_all"]
    @alleles_clean = raw_result["alleles_clean"]
    @alleles_for_mismatches = raw_result["alleles_for_mismatches"]
    @mismatches = raw_result["mismatches"]
    @ambiguous = raw_result["ambiguous"]
    @homozygous = raw_result["homozygous"]
    @locus = raw_result["locus"]
    @alg_version = raw_result["alg_version"]
    @alleles_version = raw_result["alleles_version"]
    @b5701 = raw_result["b5701"]
    @dist_b5701 = raw_result["dist_b5701"]
    @errors = raw_result["errors"]
    @all_mismatches = raw_result["all_mismatches"]
  end
end


class HLAAlgorithm
  def initialize(
    hla_std_path=nil,
    hla_freq_path=nil
  )
    @hla_std_path = hla_std_path
    @hla_freq_path = hla_freq_path
  end

  def analyze(seqs, locus='B')
    hla_input = {
      "seq1" => seqs[0],
      "seq2" => seqs[1],
      "locus" => locus,
      "hla_std_path" => nil,
      "hla_freq_path" => nil
    }

    if (!@hla_std_path.nil?)
      hla_input["hla_std_path"] = File.expand_path(@hla_std_path)
    end
    if (!@hla_freq_path.nil?)
      hla_input["hla_freq_path"] = File.expand_path(@hla_freq_path)
    end

    python_stdout, python_stderr, wait_thread = Open3.capture3(
      "#{HLA_INTERPRET_FROM_JSON} -",
      stdin_data: JSON.generate(hla_input)
    )

    if !wait_thread.success?
      error_msg = "HLA algorithm failed with exit code "\
        "#{wait_thread.value}.  Error output:\n"\
        "#{python_stderr}"
      raise error_msg
    end

    return HLAResult.new(JSON.parse(python_stdout))
  end
end
