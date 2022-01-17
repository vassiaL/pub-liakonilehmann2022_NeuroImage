# Plotting functions

using PyPlot
using Distributions

function create_correlation_figure(data, labels, prior_distributions)

    function plt_corr_vals(ax, m,n,correlation)
        ax.set_axis_bgcolor("r")
        ax.text(0.5, 0.5, "$(round(correlation[m,n],digits=3))", horizontalalignment="center", verticalalignment="center")
    end

    function plt_hist(ax, column, data)
        ax.axis("on")
        col_data = data[:,column]

        ax.hist(col_data, bins=7, density= false)
        ax.locator_params(nbins=1, axis="x")
        ax.locator_params(nbins=1, axis="y")
        ax.get_xaxis().set_visible(true)
    end

    function plt_corr_data(ax, m, n, data)
        ax.axis("on")
        ax.locator_params(nbins=2, axis="x")
        ax.locator_params(nbins=2, axis="y")
        ax.scatter(data[:,n], data[:,m])
        for s in ["top", "right"]
            ax.spines[s].set_visible(false)
        end
        # bound the x and y limits to [0,1]
        for f in [(:get_xlim, :set_xlim), (:get_ylim, :set_ylim)]
            (minval, maxval) =  ax[f[1]]()
            minval = maximum([0., minval])
            maxval = minimum([1., maxval])
            ax[f[2]]([minval, maxval])
        end
    end

    function plt_prior(ax, prior_distrib)
        # plot prior
        xl =  ax.get_xlim()
        xVals = range(xl[1], stop=xl[2], length=30)
        y = [pdf(prior_distrib,x) for x in xVals]
        ax.plot(xVals,y,linewidth=2.0, "g-")
    end

    correlation_matrix = cor(data)
    nr_cols = length(labels)+1
    fig, axes = subplots(nrows=nr_cols, ncols=nr_cols, figsize=(8,nr_cols*1.))
    for m = 1:(nr_cols)
       for n = 1:(nr_cols)
            ax = axes[m,n]
            ax.axis("off")

            if m==1 && n==1
                # nothing
            elseif m==1 && n>1
               ax.text(0.5, 0.5, "$(labels[n-1])", horizontalalignment="center", verticalalignment="baseline")
            elseif n==1 && m>1
               ax.text(0.5, 0.5, "$(labels[m-1])", horizontalalignment="right", verticalalignment="center")
            elseif m==n
                plt_hist(ax, m-1, data)
                plt_prior(ax, prior_distributions[m-1])
            elseif m>n
                plt_corr_data(ax, m-1, n-1, data)
            else
                plt_corr_vals(ax, m-1, n-1,correlation_matrix)
            end
        end
    end
    fig.tight_layout()
end



function plot_result_compact_hist(data::Array{Float64,2}, params, sup_title;
                                nr_bins = 100, add_MAP=true, plot_prior=true,
                                plot_prior_mode = true, print_MAP_in_title = true,
                                title_font_size = 6)
  plot_result(data, params, sup_title;
            nr_bins = nr_bins, add_MAP=add_MAP, plot_prior=plot_prior,
            plot_prior_mode = plot_prior_mode, nr_cols = length(params),
            print_MAP_in_title = print_MAP_in_title)
end
# ---- Plot: re-call get_MAP().
# Attention: set top_n_fraction according to what you want!
# Cause otherwise the values in the .csv files and the values in the figures
# are slightly different.
function plot_result(data::Array{Float64,2}, params, sup_title;
                    nr_bins = 50, add_MAP=true, plot_prior=true, plot_prior_mode = true,
                    nr_cols = 2, print_MAP_in_title = true, title_font_size = 10,
                    top_n_fraction = .01, conditioning_interval=0.08,
                    hist_density = true)

    nr_params = length(params)
    nr_rows = Int(ceil(nr_params/nr_cols))
    fig, axes = subplots(nrows=nr_rows, ncols=nr_cols, figsize=(9.,nr_rows*1.8))
    nr_map_samples = Int(ceil(size(data)[1] * top_n_fraction))

    top_MAP, top_n_avgMAP, top_n_min, top_n_max, top_n_stdMAP, top_n_samples = get_MAP(data, nr_map_samples)
        for i=1:nr_params

        axes[i].set_title(params[i].name,fontsize=title_font_size)
        axes[i].locator_params(nbins=4)

        (n, bins, patches) = axes[i].hist(data[:,i], nr_bins, density = hist_density, edgecolor = "none");

        plot_padding = 0.05 * (maximum(params[i].prior_Distribution) - minimum(params[i].prior_Distribution))
        axes[i].set_xlim([minimum(params[i].prior_Distribution)-plot_padding,maximum(params[i].prior_Distribution)+plot_padding])
        if plot_prior
            xl =  axes[i].get_xlim()
            min_x_prior = max(minimum(params[i].prior_Distribution), (xl[1]+plot_padding))
            max_x_prior = min(maximum(params[i].prior_Distribution), (xl[2]-plot_padding))

            xVals = range(min_x_prior, stop=max_x_prior, length=100)
            y = [pdf(params[i].prior_Distribution,x) for x in xVals]
            axes[i].plot(xVals,y,linewidth=3.0, "g-")

            if plot_prior_mode
                priorD = params[i].prior_Distribution
                priorMode = mode(priorD)
                pdfMode = pdf(priorD, priorMode)
                axes[i].plot([priorMode, priorMode],[0, pdfMode], "g--", linewidth=4.0)
            end

        end
        if add_MAP || print_MAP_in_title

            if add_MAP axes[i].plot(top_MAP[i], 0, "ro", markersize=10) end

            if print_MAP_in_title axes[i].set_title("$(params[i].name) ($(round(top_MAP[i],digits=2)))", fontsize=title_font_size) end
        end
    end
    fig.suptitle(sup_title, y=1.)
    fig.tight_layout()
    return fig
end
# ---- Plot: provide top_MAP, do not re-call get_MAP()
function plot_result(data::Array{Float64,2}, params, sup_title,
                    top_MAP;
                    nr_bins = 50, add_MAP=true, plot_prior=true, plot_prior_mode = true,
                    nr_cols = 2, print_MAP_in_title = true, title_font_size = 10,
                    top_n_fraction = .01, conditioning_interval=0.08,
                    hist_density = true)
    nr_params = length(params)
    nr_rows = Int(ceil(nr_params/nr_cols))
    fig, axes = subplots(nrows=nr_rows, ncols=nr_cols, figsize=(9.,nr_rows*1.8))
    for i=1:nr_params
        axes[i].set_title(params[i].name,fontsize=title_font_size)
        axes[i].locator_params(nbins=4)
        (n, bins, patches) = axes[i].hist(data[:,i], nr_bins, density = hist_density, edgecolor = "none");

        plot_padding = 0.05 * (maximum(params[i].prior_Distribution) - minimum(params[i].prior_Distribution))
        axes[i].set_xlim([minimum(params[i].prior_Distribution)-plot_padding,maximum(params[i].prior_Distribution)+plot_padding])
        if plot_prior
            xl =  axes[i].get_xlim()
            min_x_prior = max(minimum(params[i].prior_Distribution), (xl[1]+plot_padding))
            max_x_prior = min(maximum(params[i].prior_Distribution), (xl[2]-plot_padding))

            xVals = range(min_x_prior, stop=max_x_prior, length=100)
            y = [pdf(params[i].prior_Distribution,x) for x in xVals]
            axes[i].plot(xVals,y,linewidth=3.0, "g-")

            if plot_prior_mode
                priorD = params[i].prior_Distribution
                priorMode = mode(priorD)
                pdfMode = pdf(priorD, priorMode)
                axes[i].plot([priorMode, priorMode],[0, pdfMode], "g--", linewidth=4.0)
            end

        end
        if add_MAP || print_MAP_in_title
            if add_MAP axes[i].plot(top_MAP[i], 0, "ro", markersize=10) end
            if print_MAP_in_title axes[i].set_title("$(params[i].name) ($(round(top_MAP[i],digits=2)))", fontsize=title_font_size) end
        end
    end
    fig.suptitle(sup_title, y=1.)
    fig.tight_layout()
    return fig
end
# ---- Scatterplot LL of top samples: re-call get_MAP().
# Attention: set top_n_fraction according to what you want!
# Cause otherwise the values in the .csv files and the values in the figures
# are slightly different.
function plot_result_logLL(data::Array{Float64,2}, params, sup_title;
                        nr_cols = 2, print_MAP_in_title = true,
                        title_font_size = 10, top_n_fraction = .01
                        )
    nr_params = length(params)
    nr_rows = Int(ceil(nr_params/nr_cols))

    fig, axes = subplots(nrows=nr_rows, ncols=nr_cols, figsize=(9.,nr_rows*1.8))
    nr_map_samples = Int(ceil(size(data)[1] * top_n_fraction))

    top_MAP, top_n_avgMAP, top_n_min, top_n_max, top_n_stdMAP, top_n_samples = get_MAP(data, nr_map_samples)
        for i=1:nr_params
        axes[i].set_title(params[i].name,fontsize=title_font_size)
        axes[i].locator_params(nbins=4)
        plot_padding = 0.05 * (maximum(params[i].prior_Distribution) - minimum(params[i].prior_Distribution))
        axes[i].set_xlim([minimum(params[i].prior_Distribution)-plot_padding,maximum(params[i].prior_Distribution)+plot_padding])
        if print_MAP_in_title axes[i].set_title("$(params[i].name) ($(round(top_MAP[i],digits=2)))", fontsize=title_font_size) end

        axes[i].scatter(top_n_samples[:,i], top_n_samples[:,end], s=3)
    end
    fig.suptitle(sup_title, y=1.)
    fig.tight_layout()
    return fig
end
# ---- Scatterplot LL of top samples:
# provide top_MAP and top_n_samples, do not re-call get_MAP()
function plot_result_logLL(data::Array{Float64,2}, params, sup_title,
                        top_MAP, top_n_samples;
                        nr_cols = 2, print_MAP_in_title = true,
                        title_font_size = 10, top_n_fraction = .01
                        )

    nr_params = length(params)
    nr_rows = Int(ceil(nr_params/nr_cols))
    fig, axes = subplots(nrows=nr_rows, ncols=nr_cols, figsize=(9.,nr_rows*1.8))
        for i=1:nr_params
        axes[i].set_title(params[i].name,fontsize=title_font_size)
        axes[i].locator_params(nbins=4)
        plot_padding = 0.05 * (maximum(params[i].prior_Distribution) - minimum(params[i].prior_Distribution))
        axes[i].set_xlim([minimum(params[i].prior_Distribution)-plot_padding,maximum(params[i].prior_Distribution)+plot_padding])
        if print_MAP_in_title axes[i].set_title("$(params[i].name) ($(round(top_MAP[i],digits=2)))", fontsize=title_font_size) end

        axes[i].scatter(top_n_samples[:,i], top_n_samples[:,end], s=3)
    end
    fig.suptitle(sup_title, y=1.)
    fig.tight_layout()
    return fig
end
function plot_subj_stats(subj_stats::Step_by_step_signals, title_text_part)
        plt.figure(figsize=(9.,1.1))
       LL_subj = round(subj_stats.LL,3)
    probs = subj_stats.action_selection_prob

    legendTxt = []
    if length(subj_stats.RPE) > 0
        plt.plot(subj_stats.RPE, "xr")
        legendTxt = [legendTxt; "RPE"]
    end

    if length(subj_stats.SPE) > 0
        plt.plot(subj_stats.SPE, "ob")
        legendTxt = [legendTxt; "SPE"]
    end

    markerline, stemlines, baseline = plt.stem(probs, "g")
    legendTxt = [legendTxt; "p(a)"]

    plt.legend(legendTxt)
    meanProb = round(mean(probs),digits=3)
    plt.title("subj $(subj_stats.subject_id), $title_text_part, LL=$LL_subj, mean selection prob: $meanProb", fontsize=10)
    plt.locator_params(axis='y',nbins=2) 

end


function compare_SarsaL_RPE(data:: Array{Int64,2}, param_population, param_subject)
    println("TODO: move the function compare_SarsaL_RPE")
    LL, signalsPopulation = getLL_SarsaLambda(data, param_population[1], param_population[2], param_population[3], param_population[4], param_population[5])
    LL, signalsSubject = getLL_SarsaLambda(data, param_subject[1], param_subject[2], param_subject[3], param_subject[4], param_subject[5])

    plt.figure(figsize=(9.,1.8))
    markerline, stemlines, baseline = plt.stem(signalsPopulation.RPE, "g")
    plt.setp(markerline, "markerfacecolor", "g", "markeredgecolor", "g", "markersize", 4)
    markerline, stemlines, baseline = plt.stem(signalsSubject.RPE, "r--")
    plt.setp(markerline, "markerfacecolor", "r", "markeredgecolor", "r", "markersize", 4)
    plt.setp(baseline, "color", "k", "linewidth", 1)

    plt.legend(["RPE prior", "RPE subj"], fontsize =7)
    dot_prod= dot(signalsPopulation.RPE, signalsSubject.RPE)
    prod_norms = norm(signalsPopulation.RPE) * norm(signalsSubject.RPE)
    cos_Theta = dot_prod/prod_norms
    plt.title("cos(pop, subj)=$(round(cos_Theta,digits=3))", fontsize=10)

end
