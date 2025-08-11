old.frequencies <- read.csv(
    "src/hla_algorithm/default_data/hla_frequencies.csv",
    header=FALSE,
    col.names=c("a1", "a2", "b1", "b2", "c1", "c2"),
    colClasses="character",
)

a1.7401 <- which(old.frequencies$a1 == "7401")
a2.7401 <- which(old.frequencies$a2 == "7401")
a1.7403 <- which(old.frequencies$a1 == "7403")
a2.7403 <- which(old.frequencies$a2 == "7403")

a1.74xx <- which(startsWith(old.frequencies$a1, "74") & old.frequencies$a1 != "7400")
a2.74xx <- which(startsWith(old.frequencies$a2, "74") & old.frequencies$a2 != "7400")

old.frequencies[sort(union(a1.74xx, a2.74xx)),]

c1.17xx <- which(startsWith(old.frequencies$c1, "17") & old.frequencies$c1 != "1700")
c2.17xx <- which(startsWith(old.frequencies$c2, "17") & old.frequencies$c2 != "1700")
c1.18xx <- which(startsWith(old.frequencies$c1, "18") & old.frequencies$c1 != "1800")
c2.18xx <- which(startsWith(old.frequencies$c2, "18") & old.frequencies$c2 != "1800")

old.frequencies[sort(union(a1.74xx, a2.74xx)),]

old.frequencies[sort(union(c1.17xx, c2.17xx)),]
old.frequencies[sort(union(c1.18xx, c2.18xx)),]
