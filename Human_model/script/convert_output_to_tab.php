<?php
$file_name = $argv[1];
$fp = fopen($file_name, 'r');
$head = fgetcsv($fp);
$count = 0;
$suffix = 0;
while ($line = fgetcsv($fp)) {
    $file_data[$line[0]] = array_combine($head, $line);
    $count ++;
    if ($count >= 80) {
        $suffix ++;
        dump_file($file_name.".part.$suffix", $head, $file_data);
        $count = 0;
        $file_data = [];
    }
}
if (!empty($file_data)) {
    $suffix ++;
    dump_file($file_name.".part.$suffix", $head, $file_data);
}

dump_index_file($file_name.".index", $head);
exec("paste $file_name.index $file_name.part.* > $file_name.tab");
exec("rm $file_name.index $file_name.part.*");

function dump_index_file($filename, $head) {
    $index = array_slice($head, 1);
    $output_fp = fopen($filename, 'w+');
    fputcsv($output_fp, ['#chr', 'start', 'end'], "\t");
    foreach ($index as $h) {
        list($chr, $start, $end) = explode(".", substr($h,1));
        fputcsv($output_fp, [$chr, $start, $end], "\t");
    }
}
function dump_file($filename, $head, $file_data) {
    $index = array_slice($head, 1);
    $csv_header = [];
    foreach ($file_data as $seqId => $d) {
        $csv_header[] = "'$seqId.uniq.nodup.bam'";
    }
    $output_fp = fopen($filename, 'w+');
    fputcsv($output_fp, $csv_header, "\t");
    foreach ($index as $h) {
        $f_data = [];
        foreach ($file_data as $d) {
            $f_data[] = $d[$h];
        }

        fputcsv($output_fp, $f_data, "\t");
    }
    echo "$filename done\n";
}
var_dump(memory_get_peak_usage());
