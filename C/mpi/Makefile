username = christopher.peter.bradshaw
remote1 = ${username}@35.230.104.108
remote2 = ${username}@35.237.156.241

.PHONY: hello send_and_receive details run clean hosts

hosts:
	scp hosts ${remote1}:~

hello:
	mpicc mpi_hello.c -o hello
	scp hello ${remote1}:~/bin
	scp hello ${remote2}:~/bin

send_and_receive:
	mpicc mpi_send_and_receive.c -o send_and_receive
	scp send_and_receive ${remote1}:~/bin
	scp send_and_receive ${remote2}:~/bin

details:
	mpicc mpi_details.c -o details
	scp details ${remote1}:~/bin
	scp details ${remote2}:~/bin

clean:
	-rm -f send_and_receive hello a.out details
