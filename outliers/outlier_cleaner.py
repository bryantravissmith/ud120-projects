#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []

    ### your code goes here
    for i,value in enumerate(age):
        cleaned_data.append((abs(predictions[i]-net_worths[i]),net_worths[i],ages[i]))
    cleaned_data.sort()
    length = floor(len(cleaned_data)*0.9)

    return cleaned_data[:length]

