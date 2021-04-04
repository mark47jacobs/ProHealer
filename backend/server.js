const path = require('path');
const express = require('express');
const app = express();
const config = require('./config');
const routes = require('./routes/api');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const expressValidator = require('express-validator');
const helmet = require('helmet');
const cors = require('cors');

const server = {};

mongoose.connect(config.database, { useNewUrlParser: true, useUnifiedTopology: true },);
mongoose.Promise = global.Promise;
mongoose.connection.on('error', (err) => {
    console.log('Error in the database:', err);
});

app.use(cors());
app.use(helmet());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(expressValidator());

app.use('/api', routes);

app.use(express.static(path.join(__dirname, 'client')));

app.get('/*', (req, res) => {
    res.json('app running')
});

const PORT = process.env.PORT || config.httpPort;

app.listen(PORT, () => {
    console.log(`We have a ${config.name} server running on PORT: ${PORT}`);
});

module.exports = server;
