for pca in {8..20}; do
  sbatch-conda lucky -c 16 --mem 16G -q huge -o /dev/null -e search.error --open-mode append python -u ./script/step3_zheer_search_by_trncv.py --dt  zheer_zr1_1234 --ft zheer_zr1_1234 "$(pwd)" --npca "$pca"
  sbatch-conda lucky -c 16 --mem 16G -q huge -o /dev/null -e search.error --open-mode append python -u ./script/step3_zheer_search_by_trncv.py --dt  zheer_zr1_1234 --ft zheer_zr6_1234 "$(pwd)" --npca "$pca"
  sbatch-conda lucky -c 16 --mem 16G -q huge -o /dev/null -e search.error --open-mode append python -u ./script/step3_zheer_search_by_trncv.py --dt  zheer_zr2_1234 --ft zheer_zr2_1234 "$(pwd)" --npca "$pca"
  sbatch-conda lucky -c 16 --mem 16G -q huge -o /dev/null -e search.error --open-mode append python -u ./script/step3_zheer_search_by_trncv.py --dt  zheer_zr2_1234 --ft zheer_zr6_1234 "$(pwd)" --npca "$pca"
  sbatch-conda lucky -c 16 --mem 16G -q huge -o /dev/null -e search.error --open-mode append python -u ./script/step3_zheer_search_by_trncv.py --dt  zheer_zr11_1234 --ft zheer_zr11_1234 "$(pwd)" --npca "$pca"
  sbatch-conda lucky -c 16 --mem 16G -q huge -o /dev/null -e search.error --open-mode append python -u ./script/step3_zheer_search_by_trncv.py --dt  zheer_zr11_1234 --ft zheer_zr8_1234 "$(pwd)" --npca "$pca"
done


