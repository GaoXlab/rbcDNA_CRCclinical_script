<?php
$file_name = $argv[1];
$fp = fopen($file_name, 'r');
$head = fgetcsv($fp, null, "\t", "'");
$ids = array_map(function($v) {
    return explode('.', $v)[0];
}, array_slice($head, 3));
$seq_ids = $ids;

$info_fp = fopen("info.csv", "r");
$head = fgetcsv($info_fp);
$infos = [];
while ($line = fgetcsv($info_fp)) {
    $infos[$line[0]] = array_combine($head, $line);
}

if (in_array('r_enriched', $head)) {
    $head_array = ['r_enriched', 'r_depleted'];
} else {
    $head_array = ['rbcsum'];
}


$chunk = 1000;
$i = 0;
$file_data = [];
while ($line = fgetcsv($fp, null, "\t", "'")) {
    $pos = array_slice($line, 0, 3);
    $data = array_slice($line, 3);
    $index = "{$pos[0]}:{$pos[1]}-{$pos[2]}";
    if (empty($pos[0])) {
        continue;
    }
    $i ++;
    foreach ($seq_ids as $id) {
        $file_data[$id][$index] = array_shift($data);
    }
//    if ($i%200 ==1) var_dump($file_name . ",". $i);
    if ($i % $chunk == 0) {
        flush_data($file_data, $file_name, ceil($i / $chunk), $infos, $head_array);
        $file_data = [];
    }
}
if ($file_data) {
    flush_data($file_data, $file_name, ceil($i / $chunk), $infos,$head_array);
}
function flush_data($file_data, $file_name, $index, $infos, $head_array) {
    var_dump("Dump" . "{$file_name}.{$index}");
    $fp = fopen("{$file_name}.{$index}", 'w+');
    $head = ['seqID', 'label', ...$head_array, ...array_keys(current($file_data))];
    fputcsv($fp, $head);
    foreach ($file_data as $seqId => $d) {
        $col_data = [$seqId, $infos[$seqId]['label']];
        foreach ($head_array as $h) {
            $col_data[] = $infos[$seqId][$h];
        }
        fputcsv($fp, array_merge($col_data, $d));
    }
    fclose($fp);
}
fclose($fp);
