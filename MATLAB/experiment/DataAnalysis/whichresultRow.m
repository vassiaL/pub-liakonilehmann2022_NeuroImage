function rownumber = whichresultRow(signalname, fitType)
    resultRows = getTechEnums();
    
    if strcmp(signalname, 'RPE')
        
        if strcmp(fitType, 'pop')
            rownumber = resultRows.RPE_population;
        elseif strcmp(fitType, 'subj')
            rownumber = resultRows.RPE_subject;
        end
        
    elseif strcmp(signalname, 'SPE')
        
        if strcmp(fitType, 'pop')
            rownumber = resultRows.SPE_population;
            
        elseif strcmp(fitType, 'subj')
            rownumber = resultRows.SPE_subject;
            
        end
    elseif strcmp(signalname, 'gamma_surprise')
        
        if strcmp(fitType, 'pop')
            rownumber = resultRows.gamma_surprise_population;
            
        elseif strcmp(fitType, 'subj')
            rownumber = resultRows.gamma_surprise_subject;
            
        end
    elseif strcmp(signalname, 'action_selection_prob')
        
        if strcmp(fitType, 'pop')
            rownumber = resultRows.action_selection_prob_population;
            
        elseif strcmp(fitType, 'subj')
            rownumber = resultRows.action_selection_prob_subject;
            
        end
    elseif strcmp(signalname, 'Surprise')
        
        if strcmp(fitType, 'pop')
            rownumber = resultRows.Sbf_population;
            
        elseif strcmp(fitType, 'subj')
            rownumber = resultRows.Sbf_subject;
            
        end
    elseif strcmp(signalname, 'DeltaPolicy')
        
        if strcmp(fitType, 'pop')
            rownumber = resultRows.DeltaPolicy_population;
            
        elseif strcmp(fitType, 'subj')
            rownumber = resultRows.DeltaPolicy_subject;
            
        end
    elseif strcmp(signalname, 'StateValue')
        
        if strcmp(fitType, 'pop')
            rownumber = resultRows.StateValue_population;
            
        elseif strcmp(fitType, 'subj')
            rownumber = resultRows.StateValue_subject;
            
        end
    elseif strcmp(signalname, 'PolicyParametersDiffChosenVsUnchosen')
        
        if strcmp(fitType, 'pop')
            rownumber = resultRows.PolicyParametersDiffChosenVsUnchosen_population;
            
        elseif strcmp(fitType, 'subj')
            rownumber = resultRows.PolicyParametersDiffChosenVsUnchosen_subject;
            
        end
    else
        error('Unknown signal')
    end
        
        
end