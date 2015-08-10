@echo off
for /L %%n in (1,1,3) do (
	echo Run DepthUpsampling.exe 0 5 1 0 %%n %%n
	DepthUpsampling.exe 0 5 1 0 %%n %%n
)