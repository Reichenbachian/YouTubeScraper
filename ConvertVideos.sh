for string in $(ls out/toConvert); do
if [[ $string == *".webm"* ]]; then
	substr=$(echo $string | cut -f1 -d".")
 	ffmpeg -i "out/toConvert/$string" -c:v libx264 -c:a aac -strict experimental -b:a 192k "out/toCheck/$substr.mp4"
 	rm out/toConvert/$string
fi
done