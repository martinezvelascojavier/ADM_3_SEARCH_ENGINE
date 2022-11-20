#!/bin/bash

#First define an array of strings containing names of countries

declare -a CountryArray=("Italy" "France" "Spain" "England" "United States")

#First question:

#Select PlaceAddress column and store it in a variable
address="$(cut -f9 data.places.tsv)"

#Loop over elements on CountryArray. Print output string, grep and count occurrences 
#of the country in the address column.
for c in "${CountryArray[@]}";
do
    echo "The total number of places in $c is:"
    echo "$address" | grep -c "$c" 
    
done

#Second question

#Select NumPeople and placeAddress and store them in a variable
numpeople="$(cut -f6,9 data.places.tsv)"

#Loop over elements of CountryArray. Print output string, grep lines containing the country
#string, finally for these rows sum all the corresponding values of NumPeople and print the
#sum divided by the number of retrieved rows.
for c in "${CountryArray[@]}";
do 
    echo "The number of people who visited places in $c on average is:" 
    echo "$numpeople" | grep "$c" | awk '{avg+=$1} END {printf("%0.f\n", avg/NR)}'
done

#Third question

#Select NumPeopleWant and placeAddess and store them in a variable
numpeoplewant="$(cut -f7,9 data.places.tsv)"

#Loop over elements of CountryArray. Print output string, grep lines containing the country
#string, finally sum all the corresponding values of NumPeopleWant and print it.
for c in "${CountryArray[@]}";
do 
    echo "The total number of people who want to visit "$c" is:"
    echo "$numpeoplewant" | grep "$c" | awk '{sum+=$1} END {print(sum, '\n')}'
done

