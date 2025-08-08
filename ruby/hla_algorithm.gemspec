Gem::Specification.new do |s|
  s.name = "hla_algorithm"
  s.version = ENV['HLA_ALGORITHM_VERSION'] || '0.0.0'
  s.summary = "Ruby wrapper for the Python-based BC-CfE HLA interpretation algorithm"
  s.files = ['lib/hla_algorithm.rb']
  s.authors = [
    "Brian Wynhoven",
    "Chanson Brumme",
    "Conan Woods",
    "Rosemary McCloskey",
    "David Rickett",
    "Richard Liang"
  ]
  # Associate this gem with a GitHub repo; this allows the gems to be
  # automatically associated with the repo when pushed to GitHub Packages.
  s.metadata = {
    "github_repo" => "ssh://github.com/cfe-lab/pyeasyhla"
  }
end
