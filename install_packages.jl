using Pkg

# Add required packages
packages = [
    "DataFrames",
    "SQLite",
    "Statistics",
    "Distributions",
    "Random"
]

for package in packages
    println("Installing $package...")
    Pkg.add(package)
end

println("\nAll packages installed successfully!") 