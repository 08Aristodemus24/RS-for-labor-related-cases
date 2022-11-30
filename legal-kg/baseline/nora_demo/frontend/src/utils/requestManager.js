let RequestManager = {};

RequestManager.get = async (url) => {
    const response = await fetch(url);

    return {
        data: await response.json(),
        status: response.status
    };
};

RequestManager.validateResponse = (response) => {
    let output = {};
    let err = new Error();

    if (response.status === 200) {
        output = {...response.data};
    } else if (response.status === 401) {
        err.message = "Session expired.";
        throw err;
    } else if (response.status === 403) {
        err.message = "No permissions!";
        throw err;
    } else {
        err.message = "Server Error";
        throw err;
    }

    return output;
};

export default RequestManager;