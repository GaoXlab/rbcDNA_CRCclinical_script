INPUT=$1
NAME=$2
OUTPUT=$3

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
BASE_DIR=$(cd $SCRIPT_DIR/../modelData; pwd)

INPUT_LOCATION=`realpath $INPUT`
OUTPUT_LOCATION=`realpath $OUTPUT`
if [ ! -n "$1" ] && [ ! -n "$2" ] && [ ! -n "$3" ]; then
  echo "Parameter input_file mode_name and output filename is required"
fi
TMP_DIR=/dev/shm/$$
mkdir -p "$TMP_DIR"

cd $BASE_DIR/$NAME/cleaned || exit;
sed 's/$/.raw/' $INPUT_LOCATION | xargs -n 1 -I %1 -P 32 rsync -a ./%1 "$TMP_DIR"/
rsync -a ../sorted.tab.index "$TMP_DIR"/

cd "$TMP_DIR" || exit
sed 's/$/.raw/' $INPUT_LOCATION | xargs paste sorted.tab.index  > temp.output
mv temp.output $OUTPUT_LOCATION

rm -rf "$TMP_DIR"