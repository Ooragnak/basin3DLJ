using ColorSchemes: tol_highcontrast, tableau_10
using GLMakie
using CairoMakie
using ColorSchemes
using LaTeXStrings
using Printf
using BenchmarkTools
using Base.Threads
using Printf

simpleLaTeX() = Theme(
    fontsize = 36,
    size=(2000,1500),
    figure_padding = 48,
    palette = Attributes(color = ColorSchemes.tol_bright, linestyle = [:solid, :dash, :dot]),
    Axis = Attributes(
        xautolimitmargin = (0, 0),
        xminorticksvisible = true,
        xminorgridvisible = true,
        yminorgridvisible = true,
        yminorticksvisible = true,
        xminorticks = IntervalsBetween(4),
        yminorticks = IntervalsBetween(4),
        xticks = WilkinsonTicks(8,k_min=4,k_max=12),
        yticks = WilkinsonTicks(8,k_min=4,k_max=12),
    ),
    Band = Attributes(
        alpha = 0.1,
        cycle = [:color],
    ),
    Lines = Attributes(
        cycle = [:color, :linestyle],
    ),
)


set_theme!(merge(simpleLaTeX(),theme_latexfonts()))

# Define discrete color schemes
function lipari(n::Int)
    return get(ColorSchemes.lipari,range(0.05,0.95,n))
end

function toLaTeX(tables)
    lens = length.(tables)
    len = maximum(lens)
    table = ""
    for row in 1:1:len
        for col in 1:1:length(tables)
            if col == 1 && row <= lens[col]
                table *= prettyPrint(tables[col][row]) 
            elseif row <= lens[col]
                table *= " & " * prettyPrint(tables[col][row])  
            else
                table *= " & " 
            end
        end
        table *= " \\\\ \\hline \n"
    end
    return table
end

