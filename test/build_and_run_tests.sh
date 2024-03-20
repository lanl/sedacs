grep test_ ../mods/*.py |grep -v sdc_test | grep def |awk {'print $2'} | sed -n 's/(.*$//p' > test_list.txt

