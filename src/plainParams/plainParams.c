#include "plainParams.h"

char* plainParams(const char *executablePath, const char *get_set, const char *masterFileName, const char *masterField, const char *subField, char *value, char *resolveFlag)
{
    /**
     *      if get_set = "get" then
     *          - contents of value will be ignored and
     *          - value will contain the returned data
     *
     *      if get_set = "set" then
     *          - contents of value are used to set params file and
     *          - there is no returned data
     *
     *      Any argument = "" will be ignored.
     *
     *      To resolve paths use resolveFlag = "-r"
     *      
     */
    char command[1000];
    FILE *fp;
    char buffer[50];
    char commandLineOutput[1000] = "";
    int exit_status;
    int used_size, free_size;

    

    sprintf(command, "bash %s", executablePath);

    if (strcmp(get_set, "") != 0)
    {
        sprintf(command, "%s -a \'%s\'", command, get_set);

        if (strcmp(get_set, "get") == 0)  /* When "get", ignore value */
            strcpy(value, "");
    }

    if (strcmp(masterFileName, "") != 0)
        sprintf(command, "%s -m \'%s\'", command, masterFileName);

    if (strcmp(masterField, "") != 0)
        sprintf(command, "%s -F \'%s\'", command, masterField);

    if (strcmp(subField, "") != 0)
        sprintf(command, "%s -f \'%s\'", command, subField);

    if (strcmp(value, "") != 0)
        sprintf(command, "%s -v \'%s\'", command, value);

    if (strcmp(resolveFlag, "") != 0)
        sprintf(command, "%s %s", command, resolveFlag);

    /* Open the command for reading. */
    fp = popen(command, "r");
    if (fp == NULL)
    {
        printf("Failed to run command \"%s\"\n", command);
        exit(1);
    }

      /* Read the output a line at a time - output it. */
    while (fgets(buffer, sizeof(buffer), fp) != NULL)
    {
        used_size = sprintf(commandLineOutput, "%s%s", commandLineOutput, buffer);
        free_size = sizeof(commandLineOutput) - used_size;

        if (free_size<sizeof(buffer))
            break;
  
    }

    exit_status = pclose(fp);
    if (exit_status == 0)
    {
        strcpy(value, commandLineOutput);
        strtok(value, "\n");
    }
    else
    {
        printf("In plainParams():\n");
        printf("Error executing command \"%s\"\n", command);
        exit(1);
    }

    return value;
}